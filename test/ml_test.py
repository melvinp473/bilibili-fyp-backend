import unittest

from flask import app

from src.ml.machine_learning import *


class TestMachineLearning(unittest.TestCase):
    def setUp(self):
        self.dataset_path = "C:\\Users\\kohji\\PycharmProjects\\bilibili-fyp-backend\\MLDATA (2).csv"

    def test_linear_regression(self):
        json_data = linear_regression(self.dataset_path, ["SMOKING", "OBESITY"])
        print(json_data)
        # self.assertIsNone(self.calculator.current_datetime)
    #     self.calculator.set_current_datetime(datetime(2021, 12, 30, hour=15, minute=30))
    #     self.assertEqual(self.calculator.current_datetime, datetime(2021, 12, 30, hour=15, minute=30))
    #
    # def test_increment_current_datetime(self):
    #     self.calculator.set_current_datetime(datetime(2021, 12, 30, hour=15, minute=30))
    #     self.calculator.increment_current_datetime(timedelta(hours=20, minutes=31))
    #     self.assertEqual(self.calculator.current_datetime, datetime(2021, 12, 31, hour=12, minute=1))
    #
    # def test_timedelta_as_hour(self):
    #     self.assertAlmostEqual(self.calculator.timedelta_as_hour(timedelta(hours=5,minutes=10)),5.16666,4)
