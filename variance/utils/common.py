from os.path import join, dirname, realpath
import logging

# celltraces field separator
FIELD_SEP = '\t'

# Tor cell size
CELL_SIZE = 512

# default num instances per batch
INSTANCES_PER_BATCH = 5

# ip of the crawl instance
IP_DO = 'xxx.xxx.xxx.xxx'

# paths
BASE_DIR = dirname(dirname(realpath(__file__)))
SRC_DIR = join(BASE_DIR, 'utils')
ETC_DIR = join(BASE_DIR, 'etc')
ERRORS_DIR = join(ETC_DIR, 'visit_errors')
DATA_DIR = join(BASE_DIR, 'data')
OURS_DIR = join(DATA_DIR, 'ours')
KNN_DIR = join(OURS_DIR, 'knn')
CUMUL_DIR = join(OURS_DIR, 'cumul')
OUTPUT_DIR = join(BASE_DIR, 'output')

# files
CONFIG_FILE = join(SRC_DIR, 'config.ini')
MAPPING_FILE = join(OUTPUT_DIR, 'index_hs_domain.map')
PARAMS_FILE = join(OUTPUT_DIR, 'knn_params')
INSTANCE_MAPPING = join(OUTPUT_DIR, 'instances.map')
OUTLIERS_LIST = join(OUTPUT_DIR, 'outliers.list')

# logger
FORMAT = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
logging.basicConfig(format=FORMAT, level=logging.DEBUG)
