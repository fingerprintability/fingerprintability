from os import chdir
from os.path import isdir, basename, dirname, sep
from commands import getstatusoutput


def has_captcha(page_source):
    keywords = ['recaptcha_submit',
                'manual_recaptcha_challenge_field']
    return any(keyword in page_source for keyword in keywords)


def pack_crawl_data(crawl_dir):
    if not isdir(crawl_dir):
        print("Cannot find the crawl dir: %s" % crawl_dir)
        return False

    crawl_dir = crawl_dir[:-1] if crawl_dir.endswith(sep) else crawl_dir
    crawl_name = basename(crawl_dir)
    containing_dir = dirname(crawl_dir)
    chdir(containing_dir)
    arc_path = "%s.tar.gz" % crawl_name
    tar_cmd = "tar czvf %s %s" % (arc_path, crawl_name)
    print("Packing the crawl dir with cmd: %s" % tar_cmd)
    status, txt = getstatusoutput(tar_cmd)
    if status:
        print("Tar command failed: %s \nSt: %s txt: %s"
              % (tar_cmd, status, txt))
    else:
        # http://stackoverflow.com/a/2001749/3104416
        tar_gz_check_cmd = "gunzip -c %s | tar t > /dev/null" % arc_path
        tar_status, tar_txt = getstatusoutput(tar_gz_check_cmd)
        if tar_status:
            print("Tar check failed: %s tar_status: %s tar_txt: %s"
                  % (tar_gz_check_cmd, tar_status, tar_txt))
            return False
        else:
            return True


def run_cmd(cmd):
    return getstatusoutput('%s ' % (cmd))
