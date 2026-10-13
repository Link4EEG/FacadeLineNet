import math,unittest
import numpy as np
from facadelinenet.analysis import analyze
from facadelinenet.metrics import image_miou
class AnalysisTests(unittest.TestCase):
    def test_rectangle_geometry(self):
        mask=np.zeros((8,9),np.uint8);mask[2:5,3:7]=3
        seg,classes,s,ids=analyze(mask)
        wall=next(r for r in seg if r['class_id']==3)
        self.assertEqual(wall['area_px'],12);self.assertEqual(wall['perimeter_px'],14)
        self.assertEqual((wall['centroid_x'],wall['centroid_y']),(5,3.5))
        self.assertEqual([wall[k] for k in ['bbox_x_min','bbox_y_min','bbox_x_max','bbox_y_max']],[3,2,7,5])
        self.assertEqual(s['N_seg'],1);self.assertAlmostEqual(s['A_bar_seg'],12/72)
        self.assertEqual(s['CV_area'],0);self.assertEqual(s['PP_area'],1)
        self.assertAlmostEqual(s['C_boundary'],196/(48*math.pi))
        self.assertTrue(np.all(ids[2:5,3:7]==wall['segment_id']))
    def test_entropy_uses_class_area_not_component_count(self):
        mask=np.zeros((8,8),np.uint8);mask[1,1:5]=2;mask[3:5,1:5]=3;mask[6,1:5]=3
        _,_,s,_=analyze(mask)
        expected=-(.25*math.log2(.25)+.75*math.log2(.75))
        self.assertEqual(s['N_seg'],3);self.assertAlmostEqual(s['H_area'],expected)
        self.assertAlmostEqual(s['PP_area'],2**expected)
    def test_connectivity_and_holes(self):
        mask=np.array([[2,0],[0,2]],np.uint8)
        self.assertEqual(analyze(mask,connectivity=4)[2]['N_seg'],2)
        self.assertEqual(analyze(mask,connectivity=8)[2]['N_seg'],1)
        ring=np.full((3,3),2,np.uint8);ring[1,1]=0
        self.assertEqual(analyze(ring)[2]['L_boundary'],16)
    def test_empty(self):
        for cid in [0,1]:
            s=analyze(np.full((4,4),cid,np.uint8))[2]
            self.assertEqual(s['N_seg'],0);self.assertEqual(s['L_boundary'],0)
            for k in ['A_bar_seg','CV_area','C_boundary','H_area','PP_area']:self.assertIsNone(s[k])
    def test_normalisation(self):
        m=np.array([[2,0,2,2]],np.uint8)
        s=analyze(m,normalisation='architectural')[2]
        self.assertAlmostEqual(s['A_bar_seg'],.5)
        self.assertAlmostEqual(s['CV_area'],1/3)
    def test_metric_definition(self):
        gt=np.array([[0,0,2,255]],np.uint8);pred=np.array([[0,1,2,16]],np.uint8)
        self.assertAlmostEqual(image_miou(pred,gt),.75)
        self.assertIsNone(image_miou(pred,np.full_like(gt,255)))
    def test_invalid_labels(self):
        with self.assertRaises(ValueError):analyze(np.array([[255]],np.uint8))
if __name__=='__main__':unittest.main()
