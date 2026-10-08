"""两套绳索模型的离线验证：规格里的四个最小验证 + 单边性 + 逐环境分配。

不需要仿真器：用一个 1-D 两体积分器（机器人由速度控制器驱动、负载带滚动阻力）
把两套模型走**同一条调用路径**跑起来 —— 这同时验证了「环境层不关心用哪种模型」。

四个最小验证（对应需求里的 Test 1~4）：

1. Slack：d < L0 ⇒ T = 0；
2. Steady towing：稳定拖曳时 T_steady 应等于负载的滚动阻力，两套模型一致；
3. Slack → Taut：compliant 有明显弹性伸长、inextensible 只剩数值穿透；
   **绷直冲量（∫T dt）才是跨模型可比的不变量**（两者应几乎相等）；
4. Taut → Slack：指令归零后张力归零、负载靠惯性靠近，**不得出现压缩力**。
"""

import importlib.util
import math
from pathlib import Path
import sys
import unittest

RL = Path(__file__).resolve().parents[1]
MDP = RL / "source/imgo2_rl/imgo2_rl/tasks/manager_based/towing/mdp"
sys.path.insert(0, str(MDP))          # rope_model 的顶层回退导入需要它

try:
    import numpy as np
except ImportError:                    # pragma: no cover - 只在没装 numpy 的机器上走到
    np = None


def module_at(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


rope = module_at("towing_rope_models_rope_test", MDP / "rope.py")
models = module_at("towing_rope_models_test", MDP / "rope_model.py")
grid = module_at("towing_rope_models_grid_test", MDP / "connection_grid.py")

REST_LENGTH = 0.8
ROBOT_MASS = 12.6996                  # 整机（base 5.5339 + 四腿各 1.7914）
CART_MASS = 10.0
WHEEL_RADIUS = 0.08
WHEEL_INERTIA = 0.00128               # 绕轮轴
WHEEL_DAMPING = 0.032
DT = 0.005
DRIVE_GAIN = 200.0                    # 机器人速度控制器增益，N/(m/s)

ROBOT_INERTIA = (0.0387, 0.1041, 0.1255)      # 机体主轴系对角惯量
CART_INERTIA = (0.0787, 0.182, 0.2467)
IDENTITY = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))


def cart_effective_mass():
    return CART_MASS + 4.0 * WHEEL_INERTIA / WHEEL_RADIUS ** 2


def rolling_resistance(velocity):
    """四轮轴承阻力折到纵向：F = 4·b·ω/r，ω = v/r。"""
    return 4.0 * WHEEL_DAMPING * velocity / WHEEL_RADIUS ** 2


def body_properties(mass, inertia, offset_x):
    """名义几何：挂点与绳都在 x 轴上 ⇒ 转动项为 0（与实跑几何一致）。"""
    return models.BodyProperties(mass=mass,
                                 inverse_inertia_world=models.world_inverse_inertia(inertia,
                                                                                    IDENTITY),
                                 offset=(offset_x, 0.0, 0.0))


def make_model(name, **overrides):
    kwargs = dict(rest_length=REST_LENGTH)
    if name == "compliant":
        kwargs.update(stiffness=4000.0, damping=100.0)
    kwargs.update(overrides)
    return models.make_rope_model(name, **kwargs)


class TwoBodyRig:
    """1-D 两体：机器人由速度控制器驱动，负载带滚动阻力，中间接一个 rope model。

    挂点在 x 轴：机器人挂点在 `d`、负载挂点在 0 ⇒ 绳长就是 `d`（不是「base 间距」，
    省掉 0.41 m 的挂点偏移换算，免得把初始条件摆错）。机器人用**力**驱动
    （F = drive_gain·(v_cmd − v_R)），这样绷直冲量会按两侧质量分配。
    """

    def __init__(self, model, initial_distance, drive_gain=DRIVE_GAIN):
        self.model = model
        self.distance = initial_distance
        self.robot_velocity = 0.0
        self.cart_velocity = 0.0
        self.drive_gain = drive_gain

    def update(self, command):
        sample = self.model.update(
            robot_point=(self.distance, 0.0, 0.0), cart_point=(0.0, 0.0, 0.0),
            robot_velocity=(self.robot_velocity, 0.0, 0.0),
            cart_velocity=(self.cart_velocity, 0.0, 0.0), dt=DT,
            robot=body_properties(ROBOT_MASS, ROBOT_INERTIA, -0.16),
            cart=body_properties(cart_effective_mass(), CART_INERTIA, 0.25))
        rope_force = sample.force_on_robot[0]              # 负值 = 把机器人往回拉
        self.robot_velocity += (self.drive_gain * (command - self.robot_velocity) + rope_force) \
            / ROBOT_MASS * DT
        self.cart_velocity += (-rope_force - rolling_resistance(self.cart_velocity)) \
            / cart_effective_mass() * DT
        self.distance += (self.robot_velocity - self.cart_velocity) * DT
        return sample

    def run(self, *, command=0.0, steps=1):
        return [self.update(command) for _ in range(steps)]


class SlackTests(unittest.TestCase):
    """Test 1：松弛时两套模型都不得出力。"""

    def test_both_models_produce_no_tension_when_slack(self):
        for name in models.ROPE_MODELS:
            model = make_model(name)
            sample = model.update(robot_point=(0.4, 0, 0), cart_point=(0, 0, 0),
                                  robot_velocity=(0.5, 0, 0), cart_velocity=(0, 0, 0), dt=DT,
                                  robot=body_properties(ROBOT_MASS, ROBOT_INERTIA, -0.16),
                                  cart=body_properties(cart_effective_mass(), CART_INERTIA, 0.25))
            self.assertEqual(float(sample.rope_tension), 0.0, name)
            self.assertEqual(sample.rope_state, models.SLACK, name)
            self.assertFalse(bool(sample.is_taut), name)
            self.assertEqual(float(sample.rope_extension), 0.0, name)

    def test_approaching_never_pushes(self):
        """两挂点在靠近（ḋ<0）时即使已超长也不得产生推力——单边性是两套模型的共同底线。"""
        for name in models.ROPE_MODELS:
            model = make_model(name)
            sample = model.update(robot_point=(0.79, 0, 0), cart_point=(0, 0, 0),
                                  robot_velocity=(-1.5, 0, 0), cart_velocity=(0, 0, 0), dt=DT,
                                  robot=body_properties(ROBOT_MASS, ROBOT_INERTIA, -0.16),
                                  cart=body_properties(cart_effective_mass(), CART_INERTIA, 0.25))
            self.assertEqual(float(sample.rope_tension), 0.0, name)
            self.assertLessEqual(sample.force_on_robot[0], 0.0, name)   # 只能被拉向 −x
            self.assertGreaterEqual(sample.force_on_cart[0], 0.0, name)


class SteadyTowingTests(unittest.TestCase):
    """Test 2：稳定拖曳时张力应等于负载滚动阻力，且两套模型一致。"""

    def _steady(self, name):
        rig = TwoBodyRig(make_model(name), REST_LENGTH - 0.40)     # 与实跑一样：初始松弛 0.40 m
        samples = rig.run(command=0.5, steps=2000)                 # 10 s
        return rig, samples[-400:]                                 # 最后 2 s 当作稳态

    def test_tension_equals_rolling_resistance(self):
        tensions = {}
        for name in models.ROPE_MODELS:
            rig, tail = self._steady(name)
            tension = sum(float(s.rope_tension) for s in tail) / len(tail)
            resistance = rolling_resistance(rig.cart_velocity)
            self.assertAlmostEqual(tension, resistance, delta=0.05 * resistance + 0.05,
                                   msg=f"{name}: T={tension:.3f} vs 阻力={resistance:.3f}")
            self.assertAlmostEqual(rig.robot_velocity, rig.cart_velocity, delta=0.02, msg=name)
            tensions[name] = tension
        # 稳态张力是「换模型不该改变的量」
        self.assertAlmostEqual(tensions["compliant"], tensions["inextensible"],
                               delta=0.1 * tensions["compliant"] + 0.05)

    def test_reported_length_is_the_rest_length_plus_extension(self):
        for name in models.ROPE_MODELS:
            _, tail = self._steady(name)
            sample = tail[-1]
            self.assertAlmostEqual(float(sample.rope_length),
                                   REST_LENGTH + float(sample.rope_extension), places=9, msg=name)


class EngagementTests(unittest.TestCase):
    """Test 3：Slack → Taut。"""

    def _engage(self, name, steps=600):
        rig = TwoBodyRig(make_model(name), REST_LENGTH - 0.40)
        samples = rig.run(command=0.5, steps=steps)
        return rig, samples

    def test_compliant_stores_energy_while_inextensible_does_not(self):
        """compliant 的伸长是弹性储能（≈T/k），inextensible 只剩数值穿透（≪ 前者）。"""
        _, compliant = self._engage("compliant")
        _, inextensible = self._engage("inextensible")
        compliant_extension = max(float(s.rope_extension) for s in compliant)
        inextensible_extension = max(float(s.rope_extension) for s in inextensible)
        # 实测量级：13.1 mm（= 65 N / 4000 N/m）对 0.24 mm，差 50 倍以上
        self.assertGreater(compliant_extension, 0.010)
        self.assertLess(inextensible_extension, 0.001)
        self.assertGreater(compliant_extension / inextensible_extension, 20.0)
        # 稳态伸长必须等于 稳态张力/k（准静态时 ḋ≈0，弹簧项主导）——这是「确实是弹性伸长」的判据
        steady = compliant[-200:]
        steady_tension = sum(float(s.rope_tension) for s in steady) / len(steady)
        steady_extension = sum(float(s.rope_extension) for s in steady) / len(steady)
        self.assertAlmostEqual(steady_extension, steady_tension / 4000.0,
                               delta=0.15 * steady_extension)

    def test_total_impulse_is_the_invariant_across_models(self):
        """∫T dt 才是跨模型可比的不变量：两套模型抹掉的动量一样，所以总冲量应几乎相等。"""
        _, compliant = self._engage("compliant")
        _, inextensible = self._engage("inextensible")
        soft = sum(float(s.rope_impulse) for s in compliant)
        hard = sum(float(s.rope_impulse) for s in inextensible)
        self.assertAlmostEqual(soft, hard, delta=0.05 * soft)
        # 峰值张力是步平均量、依赖 dt 与 position_gain；这里只要求同量级
        soft_peak = max(float(s.rope_tension) for s in compliant)
        hard_peak = max(float(s.rope_tension) for s in inextensible)
        self.assertLess(hard_peak, 2.0 * soft_peak)

    def test_inextensible_keeps_the_rope_length_constraint(self):
        """约束的意义：d ≤ L0 必须真正成立（数值穿透只允许 mm 级）。"""
        rig, samples = self._engage("inextensible")
        self.assertLess(max(float(s.rope_extension) for s in samples), 0.001)
        self.assertLessEqual(rig.distance, REST_LENGTH + 0.001)
        for sample in samples:
            self.assertGreaterEqual(float(sample.rope_tension), 0.0)     # 单边
            self.assertLessEqual(float(sample.rope_length_rate), 0.5 + 1e-9)

    def test_engagement_impulse_matches_the_momentum_change(self):
        """绷直窗口内的总冲量必须与负载的动量变化 + 滚动阻力冲量平衡。

        比「单步最大冲量」稳健：`position_gain` 把绷直摊在若干步里，单步值依赖 β 与 dt，
        而**总量**由动量守恒定死。对负载列动量方程：

            J_total = m_eff·Δv_L + Σ(F_阻·dt)

        （绳只拉负载向 +x、阻力向 −x；两边都用测试自己的阻力模型，所以不是循环论证。）
        """
        rig = TwoBodyRig(make_model("inextensible"), REST_LENGTH - 0.40)
        impulse_total, resistance_total = 0.0, 0.0
        cart_speed_start, window = None, 0
        for _ in range(600):
            before = rig.cart_velocity
            sample = rig.update(0.5)
            impulse = float(sample.rope_impulse)
            if impulse > 0.0 and cart_speed_start is None:
                cart_speed_start, window = before, 1
            if window:
                impulse_total += impulse
                resistance_total += rolling_resistance(rig.cart_velocity) * DT
                window += 1
                if rig.robot_velocity - rig.cart_velocity <= 0.001 or window > 80:
                    break
        delta_momentum = cart_effective_mass() * (rig.cart_velocity - cart_speed_start)
        self.assertAlmostEqual(impulse_total, delta_momentum + resistance_total,
                               delta=0.15 * impulse_total,
                               msg=f"J={impulse_total:.4f} vs Δp+阻力冲量={delta_momentum + resistance_total:.4f}")
        # 冲量必须足以抹掉绷直时的相对速度（等效质量的量级检查）
        self.assertGreater(impulse_total, 0.5 * rig.robot_velocity / (
            1.0 / ROBOT_MASS + 1.0 / cart_effective_mass()))


class ReleaseTests(unittest.TestCase):
    """Test 4：Taut → Slack（指令归零）。"""

    def _release(self, name):
        rig = TwoBodyRig(make_model(name), REST_LENGTH - 0.40)
        rig.run(command=0.5, steps=1200)                      # 先稳定拖曳 6 s
        distance_before, speed_before = rig.distance, rig.cart_velocity
        samples = rig.run(command=0.0, steps=800)             # 指令归零 4 s
        return rig, samples, distance_before, speed_before

    def test_tension_goes_to_zero_and_never_compresses(self):
        for name in models.ROPE_MODELS:
            rig, samples, _, _ = self._release(name)
            self.assertEqual(float(samples[-1].rope_tension), 0.0, name)
            self.assertFalse(bool(samples[-1].is_taut), name)
            self.assertEqual(samples[-1].rope_state, models.SLACK, name)
            for sample in samples:
                self.assertGreaterEqual(float(sample.rope_tension), 0.0, name)
                self.assertGreaterEqual(sample.force_on_cart[0], -1e-9, name)
                self.assertLessEqual(sample.force_on_robot[0], 1e-9, name)

    def test_load_coasts_towards_the_robot_by_the_analytic_distance(self):
        """负载靠惯性继续靠近，位移应接近解析滑行距离 D = v0·τ（τ = m_eff r²/(4b)）。"""
        for name in models.ROPE_MODELS:
            rig, _, distance_before, speed_before = self._release(name)
            tau = cart_effective_mass() * WHEEL_RADIUS ** 2 / (4 * WHEEL_DAMPING)
            predicted = speed_before * tau
            travelled = distance_before - rig.distance
            self.assertAlmostEqual(travelled, predicted, delta=0.15 * predicted,
                                   msg=f"{name}: 滑行 {travelled:.4f} vs 解析 {predicted:.4f}")
            self.assertLess(rig.distance, REST_LENGTH)        # 松弛：绳被追近但不推


def _vector_body_properties(count, mass, inertia, offset_x, zeros):
    """`(N,)` 张量形态的刚体属性（逐 env 长度测试用）。"""
    return models.BodyProperties(
        mass=np.full(count, mass),
        inverse_inertia_world=models.world_inverse_inertia(
            tuple(np.full(count, value) for value in inertia),
            tuple(tuple(np.full(count, entry) for entry in row) for row in IDENTITY)),
        offset=(offset_x * np.ones(count) if not hasattr(offset_x, "shape") else offset_x,
                zeros, zeros))


class RigidLinkTests(unittest.TestCase):
    """方案 C：双边球铰连杆。与两套绳的根本差别是「可推」（反驱）。"""

    def _sample(self, distance, robot_velocity=0.0, cart_velocity=0.0,
                rest_length=REST_LENGTH):
        model = models.make_rope_model("rigid", rest_length=rest_length)
        return model.update(robot_point=(distance, 0.0, 0.0), cart_point=(0.0, 0.0, 0.0),
                            robot_velocity=(robot_velocity, 0.0, 0.0),
                            cart_velocity=(cart_velocity, 0.0, 0.0), dt=DT,
                            robot=body_properties(ROBOT_MASS, ROBOT_INERTIA, -0.16),
                            cart=body_properties(cart_effective_mass(), CART_INERTIA, 0.25))

    def test_link_is_always_engaged_and_has_no_slack(self):
        """连杆不存在松弛：任何长度下都啮合，伸长量保留符号。"""
        for distance in (REST_LENGTH, REST_LENGTH + 0.05, REST_LENGTH - 0.05):
            sample = self._sample(distance)
            self.assertTrue(bool(sample.is_taut), distance)
            self.assertEqual(sample.rope_state, models.TAUT, distance)
            self.assertAlmostEqual(float(sample.rope_extension), distance - REST_LENGTH, places=9)

    def test_rest_length_with_no_relative_motion_carries_no_force(self):
        sample = self._sample(REST_LENGTH)
        self.assertEqual(float(sample.rope_tension), 0.0)
        self.assertEqual(float(sample.rope_impulse), 0.0)

    def test_stretched_link_pulls_and_compressed_link_pushes(self):
        """双边性：拉长给拉力、压缩给推力；压缩时 `F_R` 指向机器人前方 = 反驱。"""
        stretched = self._sample(REST_LENGTH + 0.01)
        self.assertGreater(float(stretched.rope_tension), 0.0)
        self.assertLess(stretched.force_on_robot[0], 0.0)      # 被拉向 −x（朝小车）
        compressed = self._sample(REST_LENGTH - 0.01)
        self.assertLess(float(compressed.rope_tension), 0.0)   # 负张力 = 推力，绳不可能出现
        self.assertGreater(compressed.force_on_robot[0], 0.0)  # 被推向 +x = 反驱
        self.assertLess(compressed.force_on_cart[0], 0.0)      # 小车被推离机器人
        for a, b in zip(stretched.force_on_robot, stretched.force_on_cart):
            self.assertAlmostEqual(a, -b, places=12)
        for a, b in zip(compressed.force_on_robot, compressed.force_on_cart):
            self.assertAlmostEqual(a, -b, places=12)

    def test_relative_motion_along_the_link_is_resisted_both_ways(self):
        """沿连杆分离（ḋ>0）被拉回、靠拢（ḋ<0）被推开；单边绳只做前者。"""
        apart = self._sample(REST_LENGTH, robot_velocity=0.5)
        together = self._sample(REST_LENGTH, robot_velocity=-0.5)
        self.assertGreater(float(apart.rope_impulse), 0.0)
        self.assertLess(float(together.rope_impulse), 0.0)
        self.assertAlmostEqual(float(apart.rope_impulse), -float(together.rope_impulse), places=9)

    def test_rigid_link_requires_body_properties(self):
        model = models.make_rope_model("rigid", rest_length=REST_LENGTH)
        with self.assertRaises(ValueError):
            model.update(robot_point=(1.0, 0, 0), cart_point=(0, 0, 0),
                         robot_velocity=(0.5, 0, 0), cart_velocity=(0, 0, 0), dt=DT)

    def test_spawn_horizontal_gap_makes_the_3d_attachment_distance_exact(self):
        """spawn 的勾股解：h = sqrt(target² − Δz²)，再配高差正好得目标三维挂点距。

        这条几何只在上层环境 reset 时执行（需要 Isaac Lab），所以纯函数 + 这里离线覆盖；
        `upper_mdp.reset_towing_episode` 对**三类连接**统一用它把水平间距解出来
        （绳 target = 0.5·L0、刚体 target = L），避免初始约束力。
        """
        import math
        for target, delta_z in ((0.4, 0.17), (0.6, 0.17), (0.8, 0.17), (0.5, 0.0),
                                (0.2, 0.17)):
            horizontal = models.attachment_horizontal_gap(target, delta_z=delta_z)
            self.assertAlmostEqual(math.hypot(horizontal, delta_z), target, places=12)
        # 高差为 0 时就是目标距离本身；越界（target ≤ |Δz|）钳到 0，由调用方显式报错
        self.assertAlmostEqual(models.attachment_horizontal_gap(0.8, delta_z=0.0), 0.8, places=12)
        self.assertEqual(float(models.attachment_horizontal_gap(0.1, delta_z=0.2)), 0.0)

    def test_per_environment_rest_length_tensor(self):
        """逐 env 长度（训练里连杆长度按 env 随机化）：每个 env 用各自的 L。"""
        if np is None:
            self.skipTest("numpy 不在本解释器里")
        count = 3
        lengths = np.array([0.4, 0.6, 0.8])
        distance = np.full(count, 0.5)
        zeros = np.zeros(count)
        robot = _vector_body_properties(count, ROBOT_MASS, ROBOT_INERTIA, -0.16, zeros)
        cart = _vector_body_properties(count, cart_effective_mass(), CART_INERTIA, 0.25, zeros)
        model = models.make_rope_model("rigid", rest_length=lengths)
        sample = model.update(robot_point=np.stack([distance, zeros, zeros], axis=-1),
                              cart_point=np.zeros((count, 3)),
                              robot_velocity=(zeros, zeros, zeros),
                              cart_velocity=(zeros, zeros, zeros), dt=DT,
                              robot=robot, cart=cart)
        # d=0.5：L=0.4/0.6 被拉长（正张力）、L=0.8 被压缩（负张力）
        np.testing.assert_allclose(np.asarray(sample.rope_extension), 0.5 - lengths, atol=1e-12)
        self.assertGreater(float(sample.rope_tension[0]), 0.0)
        self.assertLess(float(sample.rope_tension[2]), 0.0)
        self.assertTrue(bool(np.all(np.asarray(sample.is_taut))))


class MultiRopeModelTests(unittest.TestCase):
    """三类环境逐 env 分配：按整数 id 选择，长度不同时伸长量也必须混合。"""

    def _inputs(self, count, distance):
        zeros = np.zeros(count)
        return dict(
            robot_point=np.stack([np.full(count, distance), zeros, zeros], axis=-1),
            cart_point=np.zeros((count, 3)),
            robot_velocity=(zeros, zeros, zeros), cart_velocity=(zeros, zeros, zeros), dt=DT,
            robot=_vector_body_properties(count, ROBOT_MASS, ROBOT_INERTIA, -0.16, zeros),
            cart=_vector_body_properties(count, cart_effective_mass(), CART_INERTIA, 0.25, zeros))

    def test_three_models_are_selected_per_environment(self):
        if np is None:
            self.skipTest("numpy 不在本解释器里")
        count, distance = 3, 0.45
        lengths = np.array([REST_LENGTH, REST_LENGTH, 0.5])    # 只有 rigid 用不同长度
        merged = models.MultiRopeModel(
            models=(make_model("compliant"), make_model("inextensible"),
                    models.make_rope_model("rigid", rest_length=lengths)),
            model_ids=np.array([0.0, 1.0, 2.0]))
        out = merged.update(**self._inputs(count, distance))
        # 两套绳在 0.45 < 0.8 时松弛 ⇒ 0；连杆 L=0.5 被压缩 ⇒ 负张力
        self.assertEqual(float(out.rope_tension[0]), 0.0)
        self.assertEqual(float(out.rope_tension[1]), 0.0)
        self.assertLess(float(out.rope_tension[2]), 0.0)
        # 长度不同 ⇒ 伸长量必须逐 env 混合，不能像旧的 SplitRopeModel 那样只取一套。
        # 两套绳松弛时报 0（`max(0, d−L0)`），连杆保留符号报 −0.05（压缩）——若只取
        # models[0] 的伸长量，env2 会错报 0。
        np.testing.assert_allclose(np.asarray(out.rope_extension),
                                   np.array([0.0, 0.0, distance - 0.5]), atol=1e-12)
        self.assertEqual(out.rope_state, models.MIXED)
        np.testing.assert_allclose(np.asarray(out.is_taut), np.array([0.0, 0.0, 1.0]))

    def test_multi_rope_model_needs_at_least_two_models(self):
        with self.assertRaises(ValueError):
            models.MultiRopeModel(models=(make_model("compliant"),), model_ids=np.array([0.0]))


class GridElasticIntegrationTests(unittest.TestCase):
    """场景网格新引入的弹性档必须能真的积分，而不只是通过解析上界。

    用户 2026-10-08 选的四档里最硬的是 k=1×10⁵ N/m，比历史默认 4000 大 25 倍；显式弹簧
    的稳定性由 k/c/μ 与 dt 共同决定，故用 1-D 两体积分器真跑一遍，确认不发散。
    """

    def test_stiffest_grid_level_integrates_without_diverging(self):
        stiffness, damping = grid.ELASTIC_KC[-1]
        self.assertGreaterEqual(stiffness, 100000.0)          # 确认测的是最硬档
        rig = TwoBodyRig(make_model("compliant", stiffness=stiffness, damping=damping),
                         REST_LENGTH - 0.40)
        samples = rig.run(command=0.5, steps=2000)            # 10 s
        for sample in samples:
            self.assertTrue(math.isfinite(float(sample.rope_tension)))
            self.assertTrue(math.isfinite(float(sample.rope_extension)))
            self.assertLess(abs(float(sample.rope_tension)), 1.0e5)   # 不出现爆量
        self.assertTrue(math.isfinite(rig.robot_velocity) and math.isfinite(rig.cart_velocity))
        self.assertLess(abs(rig.robot_velocity), 5.0)
        self.assertLess(abs(rig.cart_velocity), 5.0)
        # 稳态仍应把小车拖起来（张力≈滚动阻力），而不是数值噪声
        steady = samples[-400:]
        tension = sum(float(s.rope_tension) for s in steady) / len(steady)
        self.assertAlmostEqual(tension, rolling_resistance(rig.cart_velocity),
                               delta=0.15 * rolling_resistance(rig.cart_velocity) + 0.1)


class InterfaceTests(unittest.TestCase):
    """统一接口：字段一致、工厂行为、以及 compliant 与既有 rope.py 逐位一致。"""

    def test_both_models_expose_the_same_fields(self):
        self.assertEqual(set(models.RopeSample._fields),
                         {"rope_length", "rope_extension", "rope_tension", "rope_length_rate",
                          "is_taut", "rope_state", "rope_impulse", "direction",
                          "force_on_robot", "force_on_cart"})
        # `ROPE_MODELS` 的语义是「绳」——两套绳只有拉力，单边性测试可以整表套用；
        # 刚体球铰连杆是双边模型，单列在 `CONNECTION_MODELS` 里（见 RigidLinkTests）。
        self.assertEqual(set(models.ROPE_MODELS), {"compliant", "inextensible"})
        self.assertEqual(set(models.CONNECTION_MODELS),
                         {"compliant", "inextensible", "rigid"})
        self.assertEqual(models.CONNECTION_MODELS[:2], models.ROPE_MODELS)

    def test_factory_rejects_unknown_model_and_missing_parameters(self):
        with self.assertRaises(ValueError):
            models.make_rope_model("rubber_band", rest_length=REST_LENGTH)
        with self.assertRaises(ValueError):
            models.make_rope_model("compliant", rest_length=REST_LENGTH)          # 缺 k/c
        with self.assertRaises(ValueError):
            models.InextensibleRope(rest_length=REST_LENGTH, position_gain=1.5)   # 会过冲

    def test_inextensible_requires_body_properties(self):
        model = make_model("inextensible")
        with self.assertRaises(ValueError):
            model.update(robot_point=(1.0, 0, 0), cart_point=(0, 0, 0),
                         robot_velocity=(0.5, 0, 0), cart_velocity=(0, 0, 0), dt=DT)

    def test_compliant_matches_the_existing_rope_arithmetic(self):
        """compliant 模型必须与 P3/P4 已有的 rope_state 逐位一致（19 个旧 run 不能失效）。"""
        model = make_model("compliant")
        for distance, robot_velocity, cart_velocity in ((0.5, 0.5, 0.0), (0.9, 0.0, 0.0),
                                                        (1.2, -0.3, 0.1), (2.0, 0.7, -0.2)):
            sample = model.update(robot_point=(distance, 0, 0), cart_point=(0, 0, 0),
                                  robot_velocity=(robot_velocity, 0, 0),
                                  cart_velocity=(cart_velocity, 0, 0), dt=DT)
            reference = rope.rope_state((distance, 0, 0), (0, 0, 0), (robot_velocity, 0, 0),
                                        (cart_velocity, 0, 0), rest_length=REST_LENGTH,
                                        stiffness=4000.0, damping=100.0)
            self.assertAlmostEqual(float(sample.rope_tension), float(reference.tension), places=9)
            for got, want in zip(sample.force_on_robot, reference.force_on_robot):
                self.assertAlmostEqual(got, want, places=9)


class EffectiveMassTests(unittest.TestCase):
    """等效逆质量：轴向几何下转动项为零，偏置几何下必须与解析式一致。"""

    def test_axial_geometry_has_no_rotational_term(self):
        robot = body_properties(ROBOT_MASS, ROBOT_INERTIA, -0.16)
        cart = body_properties(cart_effective_mass(), CART_INERTIA, 0.25)
        inverse_mass = models.effective_inverse_mass((-1.0, 0.0, 0.0), robot, cart)
        self.assertAlmostEqual(inverse_mass, 1.0 / ROBOT_MASS + 1.0 / cart_effective_mass(),
                               places=9)

    def test_off_axis_geometry_matches_hand_computation(self):
        """挂点偏移与绳方向垂直时：k_eff = 1/m + r²/I（转动自由度让等效质量更小）。"""
        mass, inertia, arm = 2.0, (0.5, 0.5, 0.5), 0.5
        infinite_robot = models.BodyProperties(
            mass=1e12, inverse_inertia_world=models.world_inverse_inertia(inertia, IDENTITY),
            offset=(0.0, 0.0, 0.0))
        cart = models.BodyProperties(
            mass=mass, inverse_inertia_world=models.world_inverse_inertia(inertia, IDENTITY),
            offset=(arm, 0.0, 0.0))
        inverse_mass = models.effective_inverse_mass((0.0, 1.0, 0.0), infinite_robot, cart)
        self.assertAlmostEqual(inverse_mass, 1.0 / mass + arm ** 2 / 0.5, places=9)

    def test_world_inverse_inertia_rotates_the_diagonal(self):
        """`I_w⁻¹ = R diag(1/I) Rᵀ`：绕 z 转 90° 后 x/y 两个对角元互换。"""
        rotation = ((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0))
        inverse = models.world_inverse_inertia((2.0, 4.0, 8.0), rotation)
        self.assertAlmostEqual(inverse[0][0], 1.0 / 4.0, places=9)
        self.assertAlmostEqual(inverse[1][1], 1.0 / 2.0, places=9)
        self.assertAlmostEqual(inverse[2][2], 1.0 / 8.0, places=9)


@unittest.skipUnless(np is not None, "需要 numpy 才能验证批量路径")
class BatchedAndSplitTests(unittest.TestCase):
    """逐环境分配（训练时 1:1）与批量输入：同一套接口必须能吃 (N,) / (N, 3)。"""

    def _batched_inputs(self, distances):
        count = len(distances)
        zeros = np.zeros(count)
        points_robot = np.zeros((count, 3))
        points_robot[:, 0] = distances
        velocities_robot = np.zeros((count, 3))
        velocities_robot[:, 0] = 0.5
        return dict(
            robot_point=points_robot, cart_point=np.zeros((count, 3)),
            robot_velocity=velocities_robot, cart_velocity=np.zeros((count, 3)), dt=DT,
            robot=models.BodyProperties(mass=np.full(count, ROBOT_MASS),
                                        inverse_inertia_world=models.world_inverse_inertia(
                                            tuple(np.full(count, value) for value in ROBOT_INERTIA),
                                            tuple(tuple(np.full(count, entry) for entry in row)
                                                  for row in IDENTITY)),
                                        offset=(np.full(count, -0.16), zeros, zeros)),
            cart=models.BodyProperties(mass=np.full(count, cart_effective_mass()),
                                       inverse_inertia_world=models.world_inverse_inertia(
                                           tuple(np.full(count, value) for value in CART_INERTIA),
                                           tuple(tuple(np.full(count, entry) for entry in row)
                                                 for row in IDENTITY)),
                                       offset=(np.full(count, 0.25), zeros, zeros)))

    def test_point_velocity_tuple_of_arrays_is_accepted(self):
        """`point_velocity()` 返回**分量元组**；批量时元组里装的是 `(N,)` 数组。

        回归（2026-09-21 多环境）：把这种元组直接喂进模型时报
        `TypeError: robot_velocity[0] must be a real number, got Tensor` ——
        `_components` 对元组元素一律按标量校验。现在「元素是数组就原样传下去」。
        """
        vel = np.array([[0.5, 0.0, 0.0], [0.4, 0.0, 0.0]])
        omega = np.zeros((2, 3))
        offset = np.array([[-0.16, 0.0, 0.0], [-0.16, 0.0, 0.0]])
        components = rope.point_velocity(vel, omega, offset)          # 元组，元素是 (2,) 数组
        self.assertIsInstance(components, tuple)
        self.assertEqual(components[0].shape, (2,))
        sample = make_model("compliant").update(
            robot_point=np.array([[0.9, 0.0, 0.0], [0.9, 0.0, 0.0]]),
            cart_point=np.zeros((2, 3)), robot_velocity=components,
            cart_velocity=(np.zeros(2), np.zeros(2), np.zeros(2)), dt=DT)
        self.assertEqual(sample.rope_tension.shape, (2,))
        # 两个 env 的 ḋ 不同（0.5 / 0.4）⇒ 张力按 T = k·(d−L0) + c·ḋ 各自算出：
        # 4000×(0.9−0.8) + 100×ḋ = 450 / 440（逐个 env 都对，说明批量算术没错位）
        self.assertAlmostEqual(float(sample.rope_tension[0]), 450.0, places=6)
        self.assertAlmostEqual(float(sample.rope_tension[1]), 440.0, places=6)

    def test_models_run_batched_and_are_per_env_consistent(self):
        """批量跑一遍（env 0 松弛、env 1 已超长），结果必须与逐 env 单跑一致。"""
        inputs = self._batched_inputs([0.4, 0.81])
        for name in models.ROPE_MODELS:
            batched = make_model(name).update(**inputs)
            singles = [make_model(name).update(
                robot_point=(distance, 0, 0), cart_point=(0, 0, 0),
                robot_velocity=(0.5, 0, 0), cart_velocity=(0, 0, 0), dt=DT,
                robot=body_properties(ROBOT_MASS, ROBOT_INERTIA, -0.16),
                cart=body_properties(cart_effective_mass(), CART_INERTIA, 0.25))
                for distance in (0.4, 0.81)]
            for index, single in enumerate(singles):
                self.assertAlmostEqual(float(batched.rope_tension[index]),
                                       float(single.rope_tension), places=9, msg=name)
                self.assertAlmostEqual(float(batched.rope_extension[index]),
                                       float(single.rope_extension), places=9, msg=name)
                self.assertAlmostEqual(float(batched.force_on_cart[0][index]),
                                       float(single.force_on_cart[0]), places=9, msg=name)
            self.assertEqual(batched.rope_state, models.MIXED)      # 批量下无逐 env 字符串

    def test_split_model_dispatches_per_env(self):
        """1:1 分配：掩码为 1 的 env 用 inextensible、为 0 的用 compliant，逐 env 取各自结果。"""
        mask = np.array([0.0, 1.0])
        split = models.SplitRopeModel(compliant=make_model("compliant"),
                                      inextensible=make_model("inextensible"),
                                      inextensible_mask=mask)
        inputs = self._batched_inputs([0.81, 0.81])              # 都已超长，两者差异明显
        merged = split.update(**inputs)
        soft = make_model("compliant").update(**inputs)
        hard = make_model("inextensible").update(**inputs)
        self.assertAlmostEqual(float(merged.rope_tension[0]), float(soft.rope_tension[0]), places=9)
        self.assertAlmostEqual(float(merged.rope_tension[1]), float(hard.rope_tension[1]), places=9)
        self.assertAlmostEqual(float(merged.rope_impulse[0]), float(soft.rope_impulse[0]), places=9)
        self.assertAlmostEqual(float(merged.rope_impulse[1]), float(hard.rope_impulse[1]), places=9)
        # 几何量两套一致 ⇒ 直接用（不是混合）
        self.assertAlmostEqual(float(merged.rope_length[0]), float(soft.rope_length[0]), places=9)
        self.assertEqual(merged.rope_state, models.MIXED)
        self.assertTrue(bool(merged.is_taut[1]))

    def test_four_env_split_mirrors_what_the_entry_builds(self):
        """照**入口的形状**构造 4 个 env（2 compliant + 2 inextensible）跑一遍。

        这条模拟多环境可视化那条路：挂点/速度是 `(4,3)`、`point_velocity()` 给的是分量元组、
        逐 env 刚体属性是 `(4,)`/3×3 of `(4,)`、`SplitRopeModel` 用 `(4,)` 的 0/1 掩码。
        断言：① 每个 env 的输出与「单独跑那一个模型」逐位一致（掩码混合没错位）；
        ② 前两个 env 用 compliant、后两个用 inextensible（伸长量差一个量级以上）。
        这些都不需要 GPU，numpy 就能覆盖——上一次 `robot_velocity[0] must be a real number`
        就是这一层形态没被测过。
        """
        count = 4
        names = ["compliant", "compliant", "inextensible", "inextensible"]
        distance = np.full(count, REST_LENGTH + 0.01)          # 全部张紧，两模型差别明显
        robot_velocity = np.zeros((count, 3)); robot_velocity[:, 0] = 0.5
        points_robot = np.stack([distance, np.zeros(count), np.zeros(count)], axis=-1)
        zeros = np.zeros(count)
        robot = models.BodyProperties(
            mass=np.full(count, ROBOT_MASS),
            inverse_inertia_world=models.world_inverse_inertia(
                tuple(np.full(count, value) for value in ROBOT_INERTIA),
                tuple(tuple(np.full(count, entry) for entry in row) for row in IDENTITY)),
            offset=(-0.16 * np.ones(count), zeros, zeros))
        cart = models.BodyProperties(
            mass=np.full(count, cart_effective_mass()),
            inverse_inertia_world=models.world_inverse_inertia(
                tuple(np.full(count, value) for value in CART_INERTIA),
                tuple(tuple(np.full(count, entry) for entry in row) for row in IDENTITY)),
            offset=(0.25 * np.ones(count), zeros, zeros))
        # 速度走 `point_velocity()`（分量元组）——入口就是这么传的
        velocities = rope.point_velocity(robot_velocity, np.zeros((count, 3)),
                                         np.stack([-0.16 * np.ones(count), zeros, zeros], axis=-1))

        split = models.SplitRopeModel(
            compliant=make_model("compliant"), inextensible=make_model("inextensible"),
            inextensible_mask=np.array([1.0 if name == "inextensible" else 0.0 for name in names]))
        merged = split.update(robot_point=points_robot, cart_point=np.zeros((count, 3)),
                              robot_velocity=velocities,
                              cart_velocity=(zeros, zeros, zeros), dt=DT,
                              robot=robot, cart=cart)

        for index, name in enumerate(names):
            single = make_model(name).update(
                robot_point=(float(distance[index]), 0.0, 0.0), cart_point=(0.0, 0.0, 0.0),
                robot_velocity=(float(velocities[0][index]), 0.0, 0.0),
                cart_velocity=(0.0, 0.0, 0.0), dt=DT,
                robot=body_properties(ROBOT_MASS, ROBOT_INERTIA, -0.16),
                cart=body_properties(cart_effective_mass(), CART_INERTIA, 0.25))
            self.assertAlmostEqual(float(merged.rope_tension[index]), float(single.rope_tension),
                                   places=9, msg=f"env{index} ({name})")
            self.assertAlmostEqual(float(merged.rope_extension[index]),
                                   float(single.rope_extension), places=9, msg=f"env{index}")
        # 单步快照下两者的**伸长量本来就相等**（`max(0, d−L0)` 是共享几何，见 SplitRopeModel 的注释）；
        # 机制差别在张力：compliant 是 k·δ + c·ḋ = 4000×0.01 + 100×0.5 = 90 N，
        # inextensible 要一步抹掉 0.5 m/s 的相对速度 ⇒ T = J/dt = (0.5/k_eff)/dt ≈ 584 N。
        self.assertAlmostEqual(float(merged.rope_extension[0]), float(merged.rope_extension[2]), places=12)
        self.assertAlmostEqual(float(merged.rope_tension[0]), 90.0, delta=0.5)
        self.assertGreater(float(merged.rope_tension[2]), 3.0 * float(merged.rope_tension[0]),
                           "约束模型一步抹掉相对速度 ⇒ 步平均张力应明显高于弹簧")

    def test_split_model_requires_matching_rest_length(self):
        with self.assertRaises(ValueError):
            models.SplitRopeModel(compliant=make_model("compliant"),
                                  inextensible=make_model("inextensible", rest_length=1.0),
                                  inextensible_mask=np.array([0.0, 1.0]))


if __name__ == "__main__":
    unittest.main()
