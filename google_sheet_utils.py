import gspread
import pandas as pd
from oauth2client.service_account import ServiceAccountCredentials
import google_auth_oauthlib
from googleapiclient import discovery
import os

'''
Google Sheets reading utility using Sheets API   
'''

# define the scope
scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']

# add credentials to the account
creds = ServiceAccountCredentials.from_json_keyfile_name(os.path.join('ut_pr_01', 'qa_scripts', 'My Project-8f68b6c367c3.json'), scope)
service = discovery.build('sheets', 'v4', credentials=creds)
spreadsheet_id = '1JvnLy4YiCR2RqX-PSo_oV0JwS7L2XSjqXGfjpIUyHkI' #Selection Gap deliveries id
# authorize the clientsheet
client = gspread.authorize(creds)


def read_spreadsheet(sheet_name, worksheet_number):
    sheet = client.open(sheet_name)
    sheet_instance = sheet.get_worksheet(worksheet_number)
    records_data = sheet_instance.get_all_records()
    records_df = pd.DataFrame.from_dict(records_data)
    records_df = records_df[(records_df['AZ Products'] != u'') & (records_df['FK Products'] != u'')].reset_index()
    az_links_df = get_links_column_from_google_sheet(['Deliveries!F:F'], 'AZ')
    fk_links_df = get_links_column_from_google_sheet(['Deliveries!G:G'], 'FK')
    records_df = pd.concat([records_df, az_links_df, fk_links_df], axis=1)
    return records_df


def get_links_column_from_google_sheet(range, site_name):
    result = service.spreadsheets().get(spreadsheetId=spreadsheet_id, ranges=range,
                                        fields="sheets/data/rowData/values/hyperlink").execute()
    links = []
    for l in result['sheets'][0]['data'][0]['rowData']:
        if l == {}:
            continue
        links.append(l['values'][0]['hyperlink'])
    return pd.DataFrame(links, columns=['{} products link'.format(site_name)])


if __name__ == '__main__':
    read_spreadsheet('Selection Gap deliveries', 0)