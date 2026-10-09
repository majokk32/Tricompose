import json
import unittest

from tricompose_v12.radeval_image_linkage import author_path, public_record


class ImageLinkageTests(unittest.TestCase):
    def test_exact_mimic_suffix(self):
        parsed = author_path('/invented/files/p90/p90000000/s90000001/fixture-a.jpg')
        self.assertEqual(parsed['family'], 'mimic')
        self.assertEqual(parsed['relative_suffix'], 'p90/p90000000/s90000001/fixture-a.jpg')

    def test_mimic_prefix_subject_mismatch_rejected(self):
        self.assertNotEqual(author_path('/files/p91/p90000000/s90000001/a.jpg')['family'], 'mimic')

    def test_no_image_basename_retrieval(self):
        self.assertIsNone(author_path('fixture-a.jpg')['relative_suffix'])

    def test_different_file_extension_not_rewritten(self):
        parsed = author_path('/files/p90/p90000000/s90000001/a.png')
        self.assertTrue(parsed['relative_suffix'].endswith('.png'))

    def test_chexpert_requires_own_authorization(self):
        parsed = author_path('/fixture/CheXpert/train/patient900/study1/view1_frontal.jpg')
        self.assertEqual(parsed['family'], 'chexpert')
        self.assertIn('separate_access', parsed['status'])

    def test_traversal_rejected(self):
        self.assertEqual(author_path('/files/../p90/p90000000/s90000001/a.jpg')['status'], 'unsafe_path_metadata')

    def test_multiline_rejected(self):
        self.assertEqual(author_path('/a.jpg\n/b.jpg')['status'], 'unsupported_path_encoding')

    def test_nul_rejected(self):
        self.assertEqual(author_path('/a\x00.jpg')['status'], 'unsupported_path_encoding')

    def test_no_dicom_loader_claim(self):
        self.assertEqual(author_path('/files/p90/p90000000/s90000001/a.dcm')['status'], 'unsupported_image_extension')

    def test_multiple_images_not_silently_first(self):
        self.assertEqual(author_path('["a.jpg", "b.jpg"]')['family'], 'unknown')

    def test_opaque_metadata_contains_no_paths(self):
        parsed = author_path('/files/p90/p90000000/s90000001/a.jpg')
        record = public_record('source_0000', parsed, 'not_locally_available')
        text = json.dumps(record)
        self.assertNotIn('90000000', text)
        self.assertNotIn('a.jpg', text)
        self.assertFalse(record['image_bytes_read'])

    def test_patient_id_not_permitted_as_record_id(self):
        with self.assertRaises(ValueError):
            public_record('p90000000', author_path('a.jpg'), 'missing')


if __name__ == '__main__':
    unittest.main()
