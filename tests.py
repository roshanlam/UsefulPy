import unittest
from lib import *
from rich.console import Console
from rich.table import Table
from rich import box

console = Console()

class RichTestResult(unittest.TextTestResult):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.successes = []

    def addSuccess(self, test):
        super().addSuccess(test)
        self.successes.append(test)

    def printErrors(self):
        table = Table(show_header=True, header_style="bold blue", box=box.ROUNDED)
        table.add_column("Test", style="dim")
        table.add_column("Result")
        table.add_column("Details")

        for test, err in self.errors:
            table.add_row(str(test), "[red]ERROR", str(err))
        for test, err in self.failures:
            table.add_row(str(test), "[red]FAILED", str(err))
        for test in self.successes:
            table.add_row(str(test), "[green]PASSED", "-")

        console.print(table)

class TestUtilityFunctions(unittest.TestCase):

    def test_chunk_list(self):
        self.assertEqual(chunk_list([1, 2, 3, 4, 5], 2), [[1, 2], [3, 4], [5]])

    def test_flatten_list(self):
        self.assertEqual(flatten_list([[1, 2], [3, 4], [5]]), [1, 2, 3, 4, 5])

    def test_frequency_counter(self):
        self.assertEqual(frequency_counter([1, 2, 3, 3, 4]), {1: 1, 2: 1, 3: 2, 4: 1})

    def test_find_duplicates(self):
        self.assertEqual(set(find_duplicates([1, 2, 2, 3, 4, 3])), {2, 3})

    def test_capitalize_words(self):
        self.assertEqual(capitalize_words("hello world"), "Hello World")

    def test_safe_divide(self):
        self.assertEqual(safe_divide(10, 2), 5.0)
        self.assertEqual(safe_divide(10, 0), 0.0)

    def test_filter_none(self):
        self.assertEqual(filter_none([1, None, 2, None, 3]), [1, 2, 3])

    def test_deep_merge(self):
        self.assertEqual(deep_merge({'a': 1}, {'b': 2}), {'a': 1, 'b': 2})
        self.assertEqual(deep_merge({'a': {'b': 1}}, {'a': {'c': 2}}), {'a': {'b': 1, 'c': 2}})

    def test_is_palindrome(self):
        self.assertTrue(is_palindrome("radar"))
        self.assertFalse(is_palindrome("hello"))

    def test_gcd(self):
        self.assertEqual(gcd(12, 15), 3)

    def test_lcm(self):
        self.assertEqual(lcm(12, 15), 60)

    def test_group_by(self):
        self.assertEqual(group_by([{'a': 1}, {'a': 2}, {'a': 1}], 'a'), {1: [{'a': 1}, {'a': 1}], 2: [{'a': 2}]})

    def test_compact(self):
        self.assertEqual(compact([0, 1, False, 2, '', 3]), [1, 2, 3])

    def test_rotate(self):
        self.assertEqual(rotate([1, 2, 3, 4, 5], 2), [4, 5, 1, 2, 3])

    def test_binary_search(self):
        self.assertEqual(binary_search([1, 2, 3, 4, 5], 3), 2)
        self.assertEqual(binary_search([1, 2, 3, 4, 5], 6), -1)

    def test_bfs(self):
        self.assertEqual(bfs({'A': ['B', 'C'], 'B': ['D'], 'C': [], 'D': []}, 'A'), ['A', 'B', 'C', 'D'])

    def test_paginate(self):
        self.assertEqual(paginate([1, 2, 3, 4, 5], page=1, per_page=2), [1, 2])

    def test_nested_get(self):
        self.assertEqual(nested_get({'a': {'b': {'c': 1}}}, ['a', 'b', 'c']), 1)

    def test_unique(self):
        self.assertEqual(unique([1, 2, 2, 3, 4, 3]), [1, 2, 3, 4])

    def test_transpose(self):
        self.assertEqual(transpose([[1, 2], [3, 4]]), [[1, 3], [2, 4]])

    def test_ngrams(self):
        self.assertEqual(ngrams([1, 2, 3, 4], n=2), [(1, 2), (2, 3), (3, 4)])

    def test_partition(self):
        self.assertEqual(partition([1, 2, 3, 4], lambda x: x < 3), ([1, 2], [3, 4]))

    def test_mean(self):
        self.assertEqual(mean([1, 2, 3]), 2.0)

    def test_std_dev(self):
        self.assertEqual(std_dev([1, 2, 3]), 0.816496580927726)

    def test_write_file(self):
        write_file("test.txt", "hello")
        self.assertEqual(read_file("test.txt"), "hello")
        write_file("test.txt", "hello world")
        self.assertEqual(read_file("test.txt"), "hello world")

    def test_read_file(self):
        content = read_file("test.txt")
        self.assertEqual(content, "hello world")
        


if __name__ == '__main__':
    unittest.main(testRunner=unittest.TextTestRunner(resultclass=RichTestResult))