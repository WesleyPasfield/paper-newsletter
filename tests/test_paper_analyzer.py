import unittest

from functions.paper_analyzer import Paper


class PaperTests(unittest.TestCase):
    def test_equality_and_hash_use_the_same_fields(self):
        first = Paper(title='Same title', link='https://example.com/one')
        equal = Paper(title='Same title', link='https://example.com/one')
        different_link = Paper(title='Same title', link='https://example.com/two')

        self.assertEqual(first, equal)
        self.assertEqual(hash(first), hash(equal))
        self.assertNotEqual(first, different_link)


if __name__ == '__main__':
    unittest.main()
