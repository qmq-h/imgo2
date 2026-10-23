"""Offline regression for the explicit URDF/MJCF structural checker (stdlib)."""
import importlib.util
from pathlib import Path
import unittest
import xml.etree.ElementTree as ET

REPO = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('mjcf_audit', REPO/'imgo2_rl/scripts/tools/check_mjcf_current_robot.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


class StructuralAuditTest(unittest.TestCase):
    def setUp(self):
        self.urdf = ET.parse(REPO/'imgo2_description/urdf/imgo2_real.urdf').getroot()
        self.mjcf = ET.parse(REPO/'imgo2_description/mjcf/imgo2.xml').getroot()

    def test_structure_matches(self):
        result = audit.structural_audit(self.urdf, self.mjcf)
        self.assertEqual(result['issues'], [])
        self.assertEqual(result['active_joints_checked'], 12)
        self.assertEqual(result['collisions_checked'], 33)

    def test_axis_corruption_detected(self):
        self.mjcf.find('.//joint').set('axis', '0 0 1')
        self.assertTrue(any(x['field'] == 'axis' for x in audit.structural_audit(self.urdf, self.mjcf)['issues']))

    def test_origin_corruption_detected(self):
        self.mjcf.find('.//body[@name="FL_HIP"]').set('pos', '0 0 0')
        self.assertTrue(any(x['field'] == 'origin' for x in audit.structural_audit(self.urdf, self.mjcf)['issues']))

    def test_collision_corruption_detected(self):
        self.mjcf.find('.//geom[@type="box"]').set('size', '1 1 1')
        self.assertTrue(any(x['field'] == 'collision_missing_or_different' for x in audit.structural_audit(self.urdf, self.mjcf)['issues']))

    def test_effort_corruption_detected(self):
        self.mjcf.find('./actuator/motor').set('ctrlrange', '-1 1')
        self.assertTrue(any(x['field'] == 'motor_ctrlrange_or_gear' for x in audit.structural_audit(self.urdf, self.mjcf)['issues']))

    def test_collision_reordering_is_equivalent(self):
        body = self.mjcf.find('.//body[@name="base"]')
        geoms = body.findall('geom')
        for geom in geoms:
            body.remove(geom)
        for geom in reversed(geoms):
            body.append(geom)
        self.assertEqual(audit.structural_audit(self.urdf, self.mjcf)['issues'], [])

    def test_all_gazebo_inertials_and_structure_match(self):
        result = audit.audit(REPO/'imgo2_description/urdf/imgo2_real.gazebo.urdf',
                             REPO/'imgo2_description/mjcf/imgo2.xml')
        self.assertTrue(result['passed'], result)
        self.assertAlmostEqual(result['urdf_total_mass'], result['mjcf_total_mass'], places=10)
        self.assertIsNotNone(self.mjcf.find('.//body[@name="base_imu"]'))

    def test_no_extra_passive_joint_dynamics(self):
        for joint in self.mjcf.findall('.//joint'):
            self.assertEqual(float(joint.get('damping', '0')), 0)
            self.assertEqual(float(joint.get('armature', '0')), 0)
        world = ET.parse(REPO/'imgo2_deploy/ros2_ws/src/imgo2_deploy/worlds/earth.world').getroot()
        self.assertEqual(float(self.mjcf.find('option').get('timestep')),
                         float(world.find('.//physics/max_step_size').text))

    def test_gazebo_sliding_friction_wins_over_floor(self):
        source = ET.parse(REPO/'imgo2_description/urdf/imgo2_real.gazebo.urdf').getroot()
        friction = {g.get('reference'): float(g.find('mu1').text)
                    for g in source.findall('gazebo') if g.find('mu1') is not None}
        for body in self.mjcf.findall('.//body'):
            for geom in body.findall('geom'):
                if int(geom.get('contype', '1')) or int(geom.get('conaffinity', '1')):
                    self.assertEqual(float(geom.get('friction').split()[0]), friction[body.get('name')])
                    self.assertEqual(int(geom.get('priority')), 1)

    def test_sync_is_idempotent(self):
        spec = importlib.util.spec_from_file_location('mjcf_sync', REPO/'imgo2_rl/scripts/tools/sync_mjcf_from_gazebo.py')
        sync = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(sync)
        before, after = sync.synchronize(sync.DEFAULT_SOURCE, sync.DEFAULT_TARGET, sync.DEFAULT_WORLD)
        self.assertEqual(before, after)


if __name__ == '__main__':
    unittest.main()
