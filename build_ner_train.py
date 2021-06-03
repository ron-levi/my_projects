import itertools
import json
import random
from collections import Counter

import optparse

import re

import ast

import comps.attribute_extraction.extract_from_product as guys_report_module
from run.search_first import run_cmd

# import nltk
import pandas as pd

from comps.product_classifier.db_data_fetch import DataFetchFuncs
from lib.help_functions import df_to_csv, read_excel_range_to_df
from lib.time_utils import time_string
from itertools import izip, islice, tee

DONT_LIMIT = "don't limit"


def consecutive_indexes(iterable, n):
    return izip(*(islice(it, i, None) for i, it in enumerate(tee(iterable, n))))


def tuples_have_overlapping_indexes(tuples_list):
    tuples_list = sorted(tuples_list, key=lambda x: x[0])
    indexes = list(consecutive_indexes(range(len(tuples_list)), 2))
    overlapping_tuples = []
    for index_of_first, index_of_second in indexes:
        if not (tuples_list[index_of_first][1] < tuples_list[index_of_second][0]):
            overlapping_tuples.append((tuples_list[index_of_first], tuples_list[index_of_second]))
    return overlapping_tuples


def test_overlaps():
    test_ind1 = [(1, 3), (5, 7), (6, 9)]
    test_ind2 = [(1, 3), (5, 7), (8, 9)]
    test_ind3 = [(1, 5), (5, 7), (8, 9)]
    test_ind4 = [(5, 7), (1, 5), (8, 9)]
    test_ind5 = [(5, 7), (1, 3), (8, 9)]
    assert tuples_have_overlapping_indexes(test_ind1) == True
    assert tuples_have_overlapping_indexes(test_ind2) == False
    assert tuples_have_overlapping_indexes(test_ind3) == True
    assert tuples_have_overlapping_indexes(test_ind4) == True
    assert tuples_have_overlapping_indexes(test_ind5) == False


def dict_of_tuples_with_overlapping_indexes(d):
    list_of_indexes = list(itertools.chain(d.values()))
    return tuples_have_overlapping_indexes(list_of_indexes)


def map_cms_to_relevant_attribute_names():
    fn = 'ner_attributes/data/cms_to_attributes.xlsx'
    df = read_excel_range_to_df(fn, 'Sheet1', 'a1', 'k17', True)
    d = dict()
    for cms_col in df.columns:
        d[trim(cms_col)] = [trim(x) for x in list(df[cms_col]) if not pd.isnull(x)]
    return d


class AttributesIndexer(object):
    def __init__(self, cms, df, attrs_for_cms=DONT_LIMIT):
        self.cms = cms
        self.df = df
        with open('comps/attribute_extraction/function_files/attributes_dict.json') as json_file:
            self.cms_attributes_definitions_dict = {k: v for k, v in json.load(json_file).iteritems() if
                                                    attrs_for_cms == DONT_LIMIT or k in attrs_for_cms}
        self.trim_df_with_needed_columns_from_db(self.df)
        self.attributes_columns = self.get_attributes_columns(self.df)

    def run(self):
        self.df['attrs_indexes_dict'] = self.df.apply(self.annotate_row_with_indexes_of_attr_values, axis=1)
        self.remove_rows_with_overlapping_attributes()
        fn = 'ner_attributes/data/{cms}_with_indexes.csv'.format(ts=time_string(), cms=self.cms)
        df_to_csv(self.df, fn)
        return fn

    def annotate_row_with_indexes_of_attr_values(self, row):
        d = dict()
        for attr_name in self.attributes_columns:
            if pd.isna(row[attr_name]) or row[attr_name] == '':
                continue
            doc = row.product_name
            doc = doc.lower()
            possible_values_of_attribute_in_json = list(itertools.chain.from_iterable([syns_joined_with_coma.split(',')
                                                                                       for syns_joined_with_coma in
                                                                                       self.cms_attributes_definitions_dict.get(
                                                                                           attr_name, [])]))
            if not possible_values_of_attribute_in_json:
                print ('warning:', attr_name, 'not in json')
                continue
            for possible_value in possible_values_of_attribute_in_json:
                possible_value = possible_value.lower().strip()
                if possible_value == '':
                    continue

                word_delimiter_pattern = re.compile(r"\b{possible_value}\b".format(possible_value=possible_value))
                if possible_value in re.findall(word_delimiter_pattern, doc):
                    value_start_index_in_doc = re.finditer(word_delimiter_pattern, doc).next().start()
                    value_end_index_in_doc = value_start_index_in_doc + len(possible_value)
                    d[attr_name] = value_start_index_in_doc, value_end_index_in_doc
        return d

    def trim_df_with_needed_columns_from_db(self, df):
        if 'description' not in df.columns:
            self.df = DataFetchFuncs.annotate_with_product_data(df, ['description'])
        self.df = self.df.rename(columns={col: col.replace(' ', '_') for col in self.df.columns})

    @staticmethod
    def get_attributes_columns(df):
        return [col for col in df.columns if col not in ['product_id', 'product_name', 'description', 'product_type', ]]

    def remove_rows_with_overlapping_attributes(self):
        self.df['overlaps'] = self.df['attrs_indexes_dict'].apply(dict_of_tuples_with_overlapping_indexes)
        self.df = self.df[self.df['overlaps'].astype(str) == '[]']


def trim(x):
    return x.replace(' ', '_').strip().lower()


def main_create_indexed(cms, guys_report, attrs_for_cms):
    guys_report_df = pd.read_csv(guys_report, encoding='utf-8')
    if guys_report_df.empty:
        return None
    a = AttributesIndexer(cms, guys_report_df, attrs_for_cms)
    ret_fn = a.run()
    return ret_fn


def concat_all_to_one_indexed_report(cms_reports_with_index_dicts_fns):
    dfs = []
    for indexed_report_fn in cms_reports_with_index_dicts_fns:
        df = pd.read_csv(indexed_report_fn, encoding='utf-8')
        dfs.append(df)
    fn = 'ner_attributes/data/all_cms_indexed.csv'
    big_df = pd.concat(dfs)
    first_cols = [
        'product_id',
        'product_type',
        'product_name',
        'attrs_indexes_dict',
    ]
    # keys_in_all_dicts=ast.literal_eval(d_str) for d_str in big_df['attrs_indexes_dict'].values
    df = pd.DataFrame([ast.literal_eval(d_str) for d_str in big_df['attrs_indexes_dict'].values])
    big_df = big_df[first_cols + list(df.columns)]
    remaining_cols = [col for col in big_df.columns if col not in first_cols]
    big_df = big_df[first_cols + remaining_cols]
    big_df = big_df.dropna(how='all', axis='columns')

    return df_to_csv(big_df, fn)


def main():
    cms_list = [
        'allinone_desktop',
        'computer',
        'mini_pc',
        'cpu',
        'camcoder',
        'dslr_camera',
        'lens',
        'lens_cap',
        'lens_hood',

        # excluded cms:
        # 'point_shoot_camera',#empty Guy's report
        # 'tablet', error in Guy's report
    ]
    limit_cms_attributes = False
    cms_to_relevant_attribute_names = map_cms_to_relevant_attribute_names()
    cms_reports_with_index_dicts_fns = []
    for cms in cms_list:
        # 1 - Guy's report
        options = guys_report_module.process_command_line()
        options.cms_vertical_list = [cms]
        guys_report_module.main(options)
        guys_report_fn = 'ner_attributes/data/{cms}_guys_report.csv'.format(
            cms=cms)  # created in extract_from_product.py

        # 2 - create column with dictionary of indexes
        if limit_cms_attributes:
            attrs_for_cms = cms_to_relevant_attribute_names[cms]
        else:
            attrs_for_cms = DONT_LIMIT
        report_with_indexes_dict_fn = main_create_indexed(cms, guys_report_fn, attrs_for_cms)
        # report_with_indexes_dict_fn = 'ner_attributes/data/{cms}_with_indexes.csv'.format(ts=time_string(), cms=cms)

        if not report_with_indexes_dict_fn:
            print 'warning: cms', cms, 'has empty guys report'
        cms_reports_with_index_dicts_fns.append(report_with_indexes_dict_fn)

    concat_all_to_one_indexed_report(cms_reports_with_index_dicts_fns)


if __name__ == '__main__':
    main()
