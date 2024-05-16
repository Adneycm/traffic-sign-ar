import sys

from icrawler.builtin import GoogleImageCrawler

def main():
    google_crawler = GoogleImageCrawler(storage={'root_dir': 'dataset_generation/crawled/mud'})
    google_crawler.crawl(keyword='mud png', max_num=50)
    return 0

if __name__ == '__main__':
    sys.exit(main())
