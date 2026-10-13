import tempfile,unittest,json,csv
from pathlib import Path
import numpy as np
from PIL import Image
from facadelinenet.cli import main
class CliTests(unittest.TestCase):
    def test_recursive_batch_and_outputs(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);src=root/'inputs';(src/'nested').mkdir(parents=True)
            labels=np.array([[0,2,2],[1,3,3]],np.uint8)
            Image.fromarray(labels).save(src/'test.png');Image.fromarray(labels).save(src/'nested/test.png')
            out=root/'out'
            self.assertEqual(main(['--labels-only','--input',str(src),'--output',str(out)]),0)
            with (out/'facade_segmentation_analysis_results.csv').open() as f:rows=list(csv.DictReader(f))
            self.assertEqual(len(rows),2);self.assertEqual({r['image'] for r in rows},{'nested/test.png','test.png'})
            self.assertEqual(len(json.loads((out/'run_metadata.json').read_text())['images']),2)
            self.assertTrue((out/'class_ids/nested/test.png.png').is_file())
            with self.assertRaises(SystemExit):main(['--labels-only','--input',str(src),'--output',str(out)])
if __name__=='__main__':unittest.main()
