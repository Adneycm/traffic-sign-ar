import sys

from icrawler.builtin import GoogleImageCrawler

def main():
    google_crawler = GoogleImageCrawler(storage={'root_dir': 'dataset_generation/crawled/tree_branchs'})
    google_crawler.crawl(keyword='tree branchs png', max_num=25)
    return 0

if __name__ == '__main__':
    sys.exit(main())
