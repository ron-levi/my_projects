from ut_pr_01.ut import django_setup
import pandas as pd
import numpy as np
import os
import sys
import time
import os.path
import logging
from django.db import connection
from lib.customfield_utils import get_cf_def
import math
'''

'''
log_filename = '/flipkart_mnt/ron/seller_rank_tests/seller_rank_log.log'
logging.basicConfig(filename=log_filename, level=logging.DEBUG, format='%(asctime)s %(levelname)-8s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

SCORE_FUNCTION_POINTS = [(0.0, 0.0), (0.3, 0.0), (0.7, 0.3), (1.0, 1.0), (1.4, 1.3), (float("inf"), 1.3)]
NORMALIZATION_FACTOR = 1.3


def filter_df(df):
    filtered_df = df[df['passes_filters']]
    return filtered_df


def number_and_correctness_of_mskus(msku_df, weighted_scores_df, is_branded):
    msku_columns = ['product_type', 'brand'] if is_branded else ['product_type']
    msku_df = msku_df.sort_values(msku_columns, ascending=False)
    weighted_scores_df = weighted_scores_df.sort_values(msku_columns, ascending=False)
    msku_df = msku_df[msku_columns + ['cnt_fk', 'cnt_amazon']].reset_index(drop=True)
    weighted_scores_df = weighted_scores_df[msku_columns + ['cnt_fk', 'cnt_amazon']].reset_index(drop=True)
    if msku_df.equals(weighted_scores_df):
        logging.info('All MSKUs and products counts are in line between branded/unbranded_report and weighted_scores '
                     'report')
    else:
        logging.info('There is inconsistency between branded/unbranded_report and weighted_scores_report in terms of '
                     'MSKUs and products counts. Check for differences')
        # number_and_correctness_of_mskus_qa_df = pd.merge(weighted_scores_df, msku_df, on=msku_columns, how='inner',
        #                                                     suffixes=('_weighted', '_msku'))
        # return number_and_correctness_of_mskus_qa_df
    return msku_df.equals(weighted_scores_df)


def validate_ppvs_data(filtered_products_fk_df, unfiltered_fk_df, is_branded, weighted_scores_df):
    cf_defs = get_cf_def(6000, 6000, 'ppvs')

    def query_for_sum_ppvs(product_ids):
        q = '''
        select sum(cc.f{pos}) as 'sum_ppvs'
        from ut_product p
        join ut_colnumeric_collect cc on p.id = cc.product_id and cc.section_num = {section_num}
        and p.id in ({list_of_msku_product_ids}) 
        '''.format(pos=cf_defs.pos, section_num=cf_defs.section_num, list_of_msku_product_ids=product_ids)
        q_df = pd.read_sql(q, con=connection)
        return 0 if q_df.iloc[0]['sum_ppvs'] is None else int(q_df.iloc[0]['sum_ppvs'])

    filtered_products_fk_df = filtered_products_fk_df[filtered_products_fk_df['is_branded']] if is_branded else \
        filtered_products_fk_df[~filtered_products_fk_df['is_branded']]
    filtered_products_fk_df = filtered_products_fk_df.merge(unfiltered_fk_df[['product_id', 'FSN']], on=['FSN'],
                                                            how='inner')
    msku_columns = ['product_type', 'mfgr_final'] if is_branded else ['product_type']
    msku2products_filtered_fk = filtered_products_fk_df.groupby(msku_columns)['product_id'].apply(set).to_dict()
    test_ppv_df = pd.DataFrame()
    msku_columns = ['product_type', 'brand'] if is_branded else ['product_type']
    for key, value in msku2products_filtered_fk.iteritems():
        value_str_list = [str(x) for x in list(value)]
        test_sum_ppvs = query_for_sum_ppvs(",".join(value_str_list))
        if is_branded:
            row = weighted_scores_df[(weighted_scores_df['product_type'] == key[0])
                                     & (weighted_scores_df['brand'] == key[1])]
            row = row[msku_columns]
            row['test_ppvs'] = test_sum_ppvs
        else:
            row = weighted_scores_df[(weighted_scores_df['product_type'] == key)]
            row = row[msku_columns]
            row['test_ppvs'] = test_sum_ppvs
        test_ppv_df = test_ppv_df.append(row, sort=None, ignore_index=True)
    weighted_scores_df = weighted_scores_df.merge(test_ppv_df, on=msku_columns, how='outer')
    return weighted_scores_df


def validate_reviews_delta(filtered_products_az_df, unfiltered_az_df, is_branded, weighted_scores_df):
    cf_defs = get_cf_def(6000, 6002, 'reviews')

    def queries_for_reviews_delta(product_ids):
        q_curr_reviews = '''        
        select sum(cc.f{pos}) as 'current_reviews'
        from ut_product p
        join ut_colnumeric_collect cc on p.id = cc.product_id and cc.section_num = {section_num}
        and p.id in ({list_of_msku_product_ids});
        '''.format(pos=cf_defs.pos, section_num=cf_defs.section_num, list_of_msku_product_ids=product_ids)
        q_prev_reviews = '''        
        select sum(T.reviews_history) as 'reviews_history'
        from  (select p.id, min(h.value) as 'reviews_history'
                from ut_product p
                join ut_cf_history_site_6002 h on h.product_id = p.id and h.logical_name = 'reviews'
                and h.when_seen BETWEEN (UNIX_TIMESTAMP(DATE_SUB(now(), INTERVAL 3 MONTH))) 
                and (UNIX_TIMESTAMP(DATE_SUB(now(), INTERVAL 1 MONTH)))
                and p.id in ({list_of_msku_product_ids})
                group by p.id) as T;
        '''.format(list_of_msku_product_ids=product_ids)
        q_curr_df = pd.read_sql(q_curr_reviews, con=connection)
        q_prev_df = pd.read_sql(q_prev_reviews, con=connection)
        curr_reviews = 0 if q_curr_df.iloc[0]['current_reviews'] is None else q_curr_df.iloc[0]['current_reviews']
        reviews_history = 0 if q_prev_df.iloc[0]['reviews_history'] is None else q_prev_df.iloc[0]['reviews_history']
        return int(curr_reviews - reviews_history) if int(curr_reviews - reviews_history) >= 0 else int(curr_reviews)

    filtered_products_az_df = filtered_products_az_df[filtered_products_az_df['is_branded']] if is_branded else \
        filtered_products_az_df[~filtered_products_az_df['is_branded']]
    filtered_products_az_df = filtered_products_az_df.merge(unfiltered_az_df[['product_id', 'ASIN']], on=['ASIN'],
                                                            how='inner')
    msku_columns = ['product_type', 'mfgr_final'] if is_branded else ['product_type']
    msku2products_filtered_az = filtered_products_az_df.groupby(msku_columns)['product_id'].apply(set).to_dict()
    test_review_df = pd.DataFrame()
    msku_columns = ['product_type', 'brand'] if is_branded else ['product_type']
    for key, value in msku2products_filtered_az.iteritems():
        value_str_list = [str(x) for x in list(value)]
        test_reviews_delta = queries_for_reviews_delta(",".join(value_str_list))
        if is_branded:
            row = weighted_scores_df[(weighted_scores_df['product_type'] == key[0])
                                     & (weighted_scores_df['brand'] == key[1])]
            row = row[msku_columns]
            row['test_reviews_delta'] = test_reviews_delta
        else:
            row = weighted_scores_df[(weighted_scores_df['product_type'] == key)]
            row = row[msku_columns]
            row['test_reviews_delta'] = test_reviews_delta
        test_review_df = test_review_df.append(row, sort=None, ignore_index=True)
    weighted_scores_df = weighted_scores_df.merge(test_review_df, on=msku_columns, how='outer')
    return weighted_scores_df


def validate_weight_data(weighted_scores_df):
    def calc_msku_weight_test(row):
        if pd.notna(row['weight_ppvs_test']) and pd.notna(row['weight_reviews_delta_test']):
            return max(row['weight_ppvs_test'], row['weight_reviews_delta_test'])
        elif pd.notna(row['weight_ppvs_test']):
            return row['weight_ppvs_test']
        elif pd.notna(row['weight_reviews_delta_test']):
            return row['weight_reviews_delta_test']
        else:
            return 0

    LOG_BASE = 1.5
    weighted_scores_df['log_ppvs_test'] = weighted_scores_df['test_ppvs'].apply(
        lambda x: math.log(x + 1, LOG_BASE))
    weighted_scores_df['weight_ppvs_test'] = weighted_scores_df['log_ppvs_test'].apply(
        lambda val: (val / weighted_scores_df['log_ppvs_test'].sum()) * 100)
    weighted_scores_df['log_reviews_delta_test'] = weighted_scores_df['test_reviews_delta'].apply(
        lambda x: math.log(x + 1, LOG_BASE))
    weighted_scores_df['weight_reviews_delta_test'] = weighted_scores_df['log_reviews_delta_test'].apply(
        lambda val: (val / weighted_scores_df['log_reviews_delta_test'].sum()) * 100)
    weighted_scores_df['raw_weight_test'] = weighted_scores_df.apply(calc_msku_weight_test, axis=1)
    min_weight_test = min([w for w in weighted_scores_df['raw_weight_test'].to_list() if w > 0])
    weighted_scores_df['raw_weight_test'] = weighted_scores_df['raw_weight_test'].apply(
        lambda w: w if w > 0 else min_weight_test)
    weighted_scores_df['weight_test'] = weighted_scores_df['raw_weight_test'].apply(
        lambda val: (val / weighted_scores_df['raw_weight_test'].sum()) * 100)
    return weighted_scores_df


def calc_score(cnt_fk, cnt_amazon):
    if cnt_fk == 0:
        return 0
    elif cnt_amazon == 0:
        return 1
    p = 1.0 * cnt_fk / cnt_amazon
    for idx, (x, y) in enumerate(SCORE_FUNCTION_POINTS):
        if p < x:
            x_prev = SCORE_FUNCTION_POINTS[idx - 1][0]
            y_prev = SCORE_FUNCTION_POINTS[idx - 1][1]
            return (y_prev + (p - x_prev) * ((y - y_prev) / (x - x_prev))) / NORMALIZATION_FACTOR


def validate_score_data(weighted_scores_df):
    weighted_scores_df['score_test'] = weighted_scores_df.apply(
        lambda row: calc_score(row['cnt_fk'], row['cnt_amazon']), axis=1)
    return weighted_scores_df


def validate_potential_gain(weighted_scores_df):
    weighted_scores_df['potential_gain_test'] = weighted_scores_df['weight_test'] - (
            weighted_scores_df['score_test'] * weighted_scores_df['weight_test'])
    return weighted_scores_df


def validate_potential_gain_per_seller(sellers_df, weighted_scores_df, seller_potential_df, is_branded):
    if is_branded:
        sellers_df = sellers_df.sort_values(by=['site_seller_id_amazon', 'product_type', 'brand']).reset_index(
            drop=True)
        for ix, row in sellers_df.iterrows():
            weighted_row = weighted_scores_df[(weighted_scores_df['product_type'] == row['product_type']) &
                                              (weighted_scores_df['brand'] == row['brand'])]
            if len(weighted_row) == 0: print 'missing row for {0} {1}'.format(row['product_type'], row['brand'])
            try:
                curr_fk_cnt = weighted_row.iloc[0]['cnt_fk']
                curr_az_count = weighted_row.iloc[0]['cnt_amazon']
                added_fk_cnt = curr_fk_cnt + row['nonmapped_amazon']
                curr_score = calc_score(curr_fk_cnt, curr_az_count)
                potential_score = calc_score(added_fk_cnt, curr_az_count)
                potential_seller_gain = weighted_row.iloc[0]['weight'] * (potential_score - curr_score)
                sellers_df.loc[ix, 'potential_gain_per_msku_test'] = potential_seller_gain
            except IndexError as e:
                sellers_df.loc[ix, 'potential_gain_per_msku_test'] = np.nan
                print 'missing row for {0} {1}'.format(row['product_type'], row['brand'])
                logging.info('missing row for {0} {1}. \n{2}'.format(row['product_type'], row['brand'], e))
        sellers_df['potential_gain_per_msku_test'] = sellers_df.apply(lambda row: {(row['product_type'], row['brand']):
                                                                                    row['potential_gain_per_msku_test']}, axis=1)
        # sellers_df['potential_gain_per_msku_test'] = sellers_df.apply(my_func, axis=1)
        groupedby_sellers_df = sellers_df.groupby(['site_seller_id_amazon']).apply(lambda group: pd.Series(
            {'potential_gain_per_msku_test': list(group["potential_gain_per_msku_test"])})).reset_index()
        groupedby_sellers_df['potential_gain_test'] = groupedby_sellers_df['potential_gain_per_msku_test'].apply(
            lambda seller_list:
            sum(sum(item.values()) for item in seller_list))
        seller_branded_potential_df = pd.merge(seller_potential_df, groupedby_sellers_df, on=['site_seller_id_amazon'],
                                               how='inner')
        return seller_branded_potential_df
    else:
        sellers_df = sellers_df.sort_values(by=['site_seller_id_amazon', 'product_type']).reset_index(
            drop=True)
        for ix, row in sellers_df.iterrows():
            weighted_row = weighted_scores_df[(weighted_scores_df['product_type'] == row['product_type'])]
            if len(weighted_row) == 0: print 'missing row for {0} {1}'.format(row['product_type'], row['brand'])
            curr_fk_cnt = weighted_row.iloc[0]['cnt_fk']
            curr_az_count = weighted_row.iloc[0]['cnt_amazon']
            added_fk_cnt = curr_fk_cnt + row['nonmapped_amazon']
            curr_score = calc_score(curr_fk_cnt, curr_az_count)
            potential_score = calc_score(added_fk_cnt, curr_az_count)
            potential_seller_gain = weighted_row.iloc[0]['weight'] * (potential_score - curr_score)
            sellers_df.loc[ix, 'potential_gain_per_msku_test'] = potential_seller_gain
        sellers_df['potential_gain_per_msku_test'] = sellers_df.apply(lambda row: {(row['product_type']):
                                                                                       row['potential_gain_per_msku_test']}, axis=1)
        groupedby_sellers_df = sellers_df.groupby(['site_seller_id_amazon']).apply(lambda group: pd.Series(
            {'potential_gain_per_msku_test': list(group["potential_gain_per_msku_test"])})).reset_index()
        groupedby_sellers_df['potential_gain_test'] = groupedby_sellers_df['potential_gain_per_msku_test'].apply(
            lambda seller_list:
            sum(sum(item.values()) for item in seller_list))
        seller_unbranded_potential_df = pd.merge(seller_potential_df, groupedby_sellers_df,
                                                 on=['site_seller_id_amazon'], how='inner')
        return seller_unbranded_potential_df


def seller_rank_files(sellers_dir, sellers_df_with_sellers_potential_gain, filtered_products_az_df, is_branded):
    seller_files = [f for f in os.listdir(sellers_dir)]
    # Iterate over top 30 sellers in branded/unbranded directory
    for i in range(1, 31):
        seller_file = ",".join([f for f in seller_files if 'seller_rank_{0}_'.format(i) in f])
        seller_id_from_rank = sellers_df_with_sellers_potential_gain.iloc[i - 1]['site_seller_id_amazon']
        if seller_id_from_rank not in seller_file:
            logging.info('Inconsistency between order of sellers in seller_potential_gain report and order in '
                         'seller_rank files. in index {0}: seller_rank file is for seller {1} and seller_potential_gain'
                         'is for seller {2}'.format(i, seller_id_from_rank, seller_file))
        seller_file_df = pd.read_excel(sellers_dir + '/' + seller_file, encoding='utf-8')
        is_branded_filter = filtered_products_az_df['is_branded'] if is_branded else ~filtered_products_az_df['is_branded']
        seller_products = filtered_products_az_df[(is_branded_filter) & (~filtered_products_az_df['mapped'])
                                                  & (filtered_products_az_df['local_seller_ids'].str.contains(r'{0}'.format(seller_id_from_rank)))]
        seller_file_df = seller_file_df[['ASIN']].sort_values(by=['ASIN'], ascending=False)
        seller_products = seller_products[['ASIN']].sort_values(by=['ASIN'], ascending=False)
        if not seller_file_df.equals(seller_products):
            missing_seller_asins_list = list(
                set(seller_products['ASIN'].to_list()) - set(seller_file_df['ASIN'].to_list()))
            logging.info('For seller: {0} there is inconsistency between ASINs in products_AZ and seller_rank file'
                         '\nMissing ASINs are: {1}'
                         .format(seller_id_from_rank, missing_seller_asins_list))


# read all relevant reports and run tests:
def main(top_category_id, is_branded):
    selection_dir = '/flipkart_mnt/ron/seller_rank_tests/selection_gap_reports_{top_category_id}/'.format(
        top_category_id=top_category_id)
    sellers_dir = '/flipkart_mnt/ron/seller_rank_tests/weights_and_seller_ranking_{top_category_id}/'.format(
        top_category_id=top_category_id)
    for selection_file in os.listdir(selection_dir):
        if 'unfiltered_FK' in selection_file:
            unfiltered_fk_df = pd.read_csv(os.path.join(selection_dir, selection_file), encoding='utf-8')
        if 'unfiltered_AZ' in selection_file:
            unfiltered_az_df = pd.read_csv(os.path.join(selection_dir, selection_file), encoding='utf-8')
        if 'products_FK_report' in selection_file:
            products_fk_df = pd.read_csv(os.path.join(selection_dir, selection_file), encoding='utf-8')
        if 'products_AZ_report' in selection_file:
            products_az_df = pd.read_csv(os.path.join(selection_dir, selection_file), encoding='utf-8')
        if is_branded:
            if 'products_branded_report_{top_category_id}'.format(top_category_id=top_category_id) in selection_file:
                products_branded_df = pd.read_excel(os.path.join(selection_dir, selection_file), encoding='utf-8')
            if 'sellers_branded_report_{top_category_id}'.format(top_category_id=top_category_id) in selection_file:
                sellers_branded_df = pd.read_excel(os.path.join(selection_dir, selection_file), encoding='utf-8')
        else:
            if 'products_unbranded_report_{top_category_id}'.format(top_category_id=top_category_id) in selection_file:
                products_unbranded_df = pd.read_excel(os.path.join(selection_dir, selection_file), encoding='utf-8')
            if 'sellers_unbranded_report_{top_category_id}'.format(top_category_id=top_category_id) in selection_file:
                sellers_unbranded_df = pd.read_excel(os.path.join(selection_dir, selection_file), encoding='utf-8')
    sellers_dir = os.path.join(sellers_dir, 'branded') if is_branded else os.path.join(sellers_dir, 'unbranded')
    for weights_file in os.listdir(sellers_dir):
        if is_branded:
            if 'weighted_scores_branded_{top_category_id}'.format(top_category_id=top_category_id) in weights_file:
                weighted_scores_branded_df = pd.read_excel(os.path.join(sellers_dir, weights_file), encoding='utf-8')
            if 'seller_potential_gain_branded_{top_category_id}'.format(
                    top_category_id=top_category_id) in weights_file:
                seller_potential_gain_branded_df = pd.read_excel(os.path.join(sellers_dir, weights_file),
                                                               encoding='utf-8')
        else:
            if 'weighted_scores_unbranded' in weights_file:
                weighted_scores_unbranded_df = pd.read_excel(os.path.join(sellers_dir, weights_file), encoding='utf-8')
            if 'seller_potential_gain_unbranded' in weights_file:
                seller_potential_gain_unbranded_df = pd.read_excel(os.path.join(sellers_dir, weights_file),
                                                                 encoding='utf-8')
    filtered_products_fk_df = filter_df(products_fk_df)
    filtered_products_az_df = filter_df(products_az_df)
    if is_branded:
        number_and_correctness_of_mskus(products_branded_df, weighted_scores_branded_df, is_branded)
        weighted_branded_scores_with_ppvs_test = validate_ppvs_data(filtered_products_fk_df, unfiltered_fk_df,
                                                                    is_branded, weighted_scores_branded_df)
        weighted_branded_scores_with_reviews_delta_test = validate_reviews_delta(filtered_products_az_df,
                                                                                 unfiltered_az_df,
                                                                                 is_branded,
                                                                                 weighted_branded_scores_with_ppvs_test)
        weighted_branded_scores_with_weight_test = validate_weight_data(weighted_branded_scores_with_reviews_delta_test)
        weighted_branded_scores_with_scores_test = validate_score_data(weighted_branded_scores_with_weight_test)
        weighted_branded_scores_with_potential_gain_test = validate_potential_gain(
            weighted_branded_scores_with_scores_test)
        weighted_branded_scores_with_potential_gain_test.to_csv(
            sellers_dir + '/weighted_scores_branded_test_{top_category_id}.csv'.format(top_category_id=top_category_id),
            encoding='utf-8', header=True, index=False)
        # now validate seller_potential_gain report
        # weighted_branded_scores_with_potential_gain_test = pd.read_csv(os.path.join(sellers_dir, 'weighted_scores_branded_test_2035.csv'), encoding='utf-8')
        sellers_df_with_sellers_potential_gain_branded = validate_potential_gain_per_seller(sellers_branded_df,
                                                                                            weighted_branded_scores_with_potential_gain_test,
                                                                                            seller_potential_gain_branded_df,
                                                                                            is_branded)
        sellers_df_with_sellers_potential_gain_branded.to_csv(
            sellers_dir + '/seller_potential_gain_branded_test_{top_category_id}.csv'.format(
                top_category_id=top_category_id),
            encoding='utf-8', header=True, index=False)
        # sellers_df_with_sellers_potential_gain_branded = pd.read_csv(os.path.join(sellers_dir, 'seller_potential_gain_branded_test_2035.csv'), encoding='utf-8')
        seller_rank_files(sellers_dir, sellers_df_with_sellers_potential_gain_branded, filtered_products_az_df,
                          is_branded)
    else:
        number_and_correctness_of_mskus(products_unbranded_df, weighted_scores_unbranded_df, is_branded)
        weighted_unbranded_scores_with_ppvs_test = validate_ppvs_data(filtered_products_fk_df, unfiltered_fk_df,
                                                                      is_branded, weighted_scores_unbranded_df)
        weighted_unbranded_scores_with_reviews_delta_test = validate_reviews_delta(filtered_products_az_df,
                                                                                   unfiltered_az_df,
                                                                                   is_branded,
                                                                                   weighted_unbranded_scores_with_ppvs_test)
        weighted_unbranded_scores_with_weight_test = validate_weight_data(
            weighted_unbranded_scores_with_reviews_delta_test)
        weighted_unbranded_scores_with_scores_test = validate_score_data(weighted_unbranded_scores_with_weight_test)
        weighted_unbranded_scores_with_potential_gain_test = validate_potential_gain(
            weighted_unbranded_scores_with_scores_test)
        weighted_unbranded_scores_with_potential_gain_test.to_csv(
            sellers_dir + '/weighted_scores_unbranded_test_{top_category_id}.csv'.format(
                top_category_id=top_category_id),
            encoding='utf-8', header=True, index=False)
        # now validate seller_potential_gain report
        # weighted_unbranded_scores_with_potential_gain_test = pd.read_csv(os.path.join(sellers_dir, 'weighted_branded_scores_qa.csv'), encoding='utf-8')
        sellers_df_with_sellers_potential_gain_unbranded = validate_potential_gain_per_seller(sellers_unbranded_df,
                                                                                              weighted_unbranded_scores_with_potential_gain_test,
                                                                                              seller_potential_gain_unbranded_df,
                                                                                              is_branded)
        sellers_df_with_sellers_potential_gain_unbranded.to_csv(
            sellers_dir + '/seller_potential_gain_unbranded_test_{top_category_id}.csv'.format(
                top_category_id=top_category_id),
            encoding='utf-8', header=True, index=False)
        seller_rank_files(sellers_dir, sellers_df_with_sellers_potential_gain_unbranded, filtered_products_az_df,
                          is_branded)


if __name__ == '__main__':
    main(2035, is_branded=True)
    main(2035, is_branded=False)
