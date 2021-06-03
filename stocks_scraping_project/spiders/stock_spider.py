from abc import ABC
import scrapy
from collections import OrderedDict


class StockSpider(scrapy.Spider, ABC):
    name = 'stock spider'

    def __init__(self, stock_symbol='amzn', **kwargs):
        super(StockSpider, self).__init__(**kwargs)
        self.headers = {
            "accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,"
                      "application/signed-exchange;v=b3;q=0.9",
            "accept-encoding": "gzip, deflate, br",
            "accept-language": "en-GB,en;q=0.9,en-US;q=0.8,ml;q=0.7",
            "cache-control": "max-age=0",
            "dnt": "1",
            "sec-fetch-dest": "document",
            "sec-fetch-mode": "navigate",
            "sec-fetch-site": "none",
            "sec-fetch-user": "?1",
            "upgrade-insecure-requests": "1",
            "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
                          "Chrome/81.0.4044.122 Safari/537.36"}
        if stock_symbol:
            self.start_urls = [
                'https://finance.yahoo.com/quote/{stock_symbol}?p={stock_symbol}'.format(stock_symbol=stock_symbol)]
        else:
            raise NameError("Need to specify a symbol for the stock")

    def start_requests(self):
        yield scrapy.Request(self.start_urls[0], callback=self.parse_main_stock_price, headers=self.headers)
        yield scrapy.Request(self.start_urls[0] + 'historical', callback=self.parse_history_stock_info)
        yield scrapy.Request(self.start_urls[0] + 'earnings', callback=self.parse_earnings_stock_info)

    def parse_main_stock_price(self, response):
        summary_table = response.xpath('//div[contains(@data-test,"summary-table")]//tr').getall()
        table_data = OrderedDict()
        for entry in summary_table:
            key = entry.xpath('.//td[1]//text()')[0]
            value = entry.xpath('.//td[2]//text()')[0]
            table_data.update({key: value})
        yield table_data

    def parse_history_stock_info(self, response):
        pass

    def parse_earnings_stock_info(self, response):
        pass
