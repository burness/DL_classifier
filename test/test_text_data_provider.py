import unittest
import sys
sys.path.append("../data_reader/text_data")
from text_data_provider import TextReader

class TextReaderTestCase(unittest.TestCase):
    def setUp(self):
        self.text_reader = TextReader(text_data_path="./test_data/test_text_data.txt", batch_size = 4)

    def test_batch_iter(self):
        batch_num, data_gen = self.text_reader.batch_iter(shuffle=False)
        x,y = data_gen.next()
        print x,y 
        x1, y1 = data_gen.next()
        print x1,y1
        self.assertEqual(batch_num, 4)
    
    def test_data_path(self):
        self.assertEqual(self.text_reader.text_data_path, "./test_data/test_text_data.txt")


    

if __name__ == '__main__':
    unittest.main()