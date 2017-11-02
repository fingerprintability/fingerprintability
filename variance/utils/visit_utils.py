import sys
from shutil import copy2
from os.path import join, isfile, dirname, realpath, basename, abspath
sys.path.insert(0, dirname(dirname(dirname(realpath(__file__)))))

from variance import common as cm
from variance.utils.crawl_data import (is_moz_error_txt,
                                       gen_find_visit_dirs,
                                       get_page_title_from_html_src)


# USAGE:
# Check visits for error by:
# python visit_utils.py -i CRAWL_DIR -t check_errors > path/to/crawl_name_errors.txt

DEBUG = False


def check_err_txt(err_file):
    """Error lines shouldn't be longer than 27 chars
    e.g. 8_hackpdatf4kryh54.onion_81

    We use to have multiple lines printed into one when we use
    multiprocessing. We shouldn't have these errors anymore as
    we don't parallelize."""
    MAX_VISIT_DIRNAME_LEN = 27
    for line in open(err_file):
        assert len(line.rstrip()) <= MAX_VISIT_DIRNAME_LEN, line
    print err_file, "seems OK"


ETC_DIR = join(dirname(dirname(realpath(__file__))), "etc", )


def copy_errors_to_tmp(screenshot_file, visit_dir):
    copy2(screenshot_file, join(ETC_DIR,
                                "visit_errors",
                                "title_err_pngs",
                                "%s.png" % basename(visit_dir)))


def is_visit_ok(visit_dir):
    page_title = ""
    non_empty_html = False
    ff_log_file = join(visit_dir, cm.FF_LOG_FILENAME)
    http_log_file = join(visit_dir, cm.HTTP_LOG_FILENAME)
    html_src_file = join(visit_dir, cm.HTML_SRC_FILENAME)
    screenshot_file = join(visit_dir, cm.SCREENSHOT_FILENAME)
    pcap_file = join(visit_dir, cm.PCAP_FILENAME)
    files_ok = (isfile(html_src_file) and isfile(http_log_file) and
                isfile(pcap_file) and isfile(screenshot_file) and
                isfile(ff_log_file))

    if files_ok:
        html_src = open(html_src_file).read()
        non_empty_html = bool(len(html_src))
        if DEBUG and not non_empty_html:
            print "Empty page", html_src_file

        fx_conn_error = is_moz_error_txt(html_src)
        if not fx_conn_error:
            page_title = get_page_title_from_html_src(html_src)
            if DEBUG and "error" in page_title.lower():
                print "Page title error", html_src_file
                copy_errors_to_tmp(screenshot_file, visit_dir)

    # It's an error if one of the following is true:
    # - One or more of the visit files is missing
    # - there was a connection error (Firefox connection error page)
    # - HTML source is empty
    # - page title includes the word "error" (manually verified)
    is_ok = (files_ok and (not fx_conn_error) and
             non_empty_html and "error" not in page_title.lower())
    if not is_ok:
        print basename(visit_dir)  # this will be redirected to the error.txt
    return is_ok


def check_all_visits_for_error(crawl_dir):
    visit_dirs = gen_find_visit_dirs(crawl_dir)
    for visit_dir in visit_dirs:
        is_visit_ok(visit_dir)


if __name__ == '__main__':
    args = sys.argv[1:]
    in_arg = None   # input dir or file
    task = None
    if not args:
        print 'usage: -i input_dir -t task'
        sys.exit(1)

    if len(args) > 1 and args[0] == '-i':
        in_arg = abspath(args[1])
        del args[0:2]
    else:
        print "Can't understand the input dir/file", in_arg
        sys.exit(1)

    if len(args) > 1 and args[0] == '-t':
        task = args[1]
        del args[0:2]

    if task == "check_errors":
        check_all_visits_for_error(in_arg)
    elif task == "check_errors_txt":
        check_err_txt(in_arg)
    else:
        print "Can't understand the task", task
        sys.exit(1)
