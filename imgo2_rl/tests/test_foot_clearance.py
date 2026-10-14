import importlib.util
import math
from pathlib import Path
import sys
import unittest

TOOLS = Path(__file__).resolve().parents[1] / "scripts/tools"
sys.path.insert(0, str(TOOLS))
import summarize_foot_clearance as sole


class ClearanceTests(unittest.TestCase):
    def test_sphere_surface_is_not_link_origin(self):
        sphere = [([0,0,-.02], sole.fk.IDENTITY, "sphere", {"radius":(.022,)})]
        self.assertAlmostEqual(sole.collision_bottom([0,0,.042],sole.fk.IDENTITY,sphere),0)
        self.assertAlmostEqual(sole.collision_bottom([0,0,.062],sole.fk.IDENTITY,sphere),.02)

    def test_cylinder_support_changes_with_orientation(self):
        cylinder=[([0,0,0],sole.fk.IDENTITY,"cylinder",{"radius":(.02,),"length":(.01,)})]
        self.assertAlmostEqual(sole.collision_bottom([0,0,.02],sole.fk.IDENTITY,cylinder),.015)
        side=sole.rpy_rotation([0,math.pi/2,0])
        self.assertAlmostEqual(sole.collision_bottom([0,0,.02],side,cylinder),0)

    def test_all_foot_collisions_participate_and_ground_skimming_has_no_bouts(self):
        shapes=[([0,0,0],sole.fk.IDENTITY,"sphere",{"radius":(.02,)}),
                ([0,0,-.02],sole.fk.IDENTITY,"sphere",{"radius":(.022,)})]
        self.assertAlmostEqual(sole.collision_bottom([0,0,.042],sole.fk.IDENTITY,shapes),0)
        times=[i*.01 for i in range(100)]
        result=sole.height_summary(times,[.0005]*100)
        self.assertEqual(result["above_1cm_fraction"],0)
        self.assertEqual(result["height_bouts_over_5mm"],0)
        self.assertIsNone(result["height_bout_median_period_s"])

    def test_real_fk_position_and_rotation_agree_with_existing_reference(self):
        import xml.etree.ElementTree as ET
        root=ET.parse(TOOLS.parents[2]/"imgo2_description/urdf/imgo2_real.urdf").getroot()
        chain=sole.fk.read_chain(root,"FL_FOOT")
        angles=[.1,1.0,-2.0]
        position,rotation=sole.pose(chain,angles)
        self.assertEqual(position,sole.fk.forward_kinematics(chain,angles))
        for row in rotation:
            self.assertAlmostEqual(sum(x*x for x in row),1)


if __name__=="__main__":
    unittest.main()
