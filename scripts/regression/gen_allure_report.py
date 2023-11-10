# !/usr/bin/env python3
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python

import os
import time
import platform
import subprocess
import zipfile
import stat
import wget
import tarfile

def gen_allure_report():
    """
    generate allure report
    """   
    # install allure     
    exit_code, output = subprocess.getstatusoutput("allure --version")
    if exit_code == 0:
        print("allure version is:{}".format(output))
        allure_bin = "allure"
    else:
        sysstr = platform.system()
        if os.path.exists("allure-2.19.0.zip") is False:
            bin_src = "https://xly-devops.bj.bcebos.com/tools/allure-2.19.0.zip"
            bin_file = wget.download(bin_src)
            zip_file = zipfile.ZipFile(bin_file, "a")
            zip_file.extractall()
        allure_bin_f = "%s/allure-2.19.0/bin/allure" % (os.getcwd())
        st = os.stat(allure_bin_f)
        os.chmod(allure_bin_f, st.st_mode | stat.S_IEXEC)
        allure_bin = "%s/allure-2.19.0/bin/allure" % (os.getcwd())
        exit_code, output = subprocess.getstatusoutput("java -version")
        if exit_code == 0:
            print("java version is:{}".format(output))
        else: # install java
            if os.path.exists("java_linux.tar.gz") is False:
                java_src = "https://paddle-qa.bj.bcebos.com/java/java_linux.tar.gz"
                java_file = wget.download(java_src)
                tf = tarfile.open("java_linux.tar.gz")
                tf.extractall(os.getcwd())
            os.environ['JAVA_HOME'] = os.path.join(os.getcwd(), "jdk1.8.0_351")
            os.environ['JRE_HOME'] = os.path.join(os.getenv('JAVA_HOME'), "jre")
            os.environ['CLASSPATH'] = os.path.join(os.getenv('JAVA_HOME'), "lib")
            os.environ["PATH"] += os.pathsep + os.path.join(os.getenv('JAVA_HOME'), "bin")
            exit_code, output = subprocess.getstatusoutput("java -version")
            print("java version is:{}".format(output))
    exit_code, output = subprocess.getstatusoutput("%s --version" % allure_bin)
    if exit_code == 0:
        print("allure version is:{}".format(output))
        cmd = "%s generate result -o report" % allure_bin
        ret = os.system(cmd)
        if ret:
            print("allure generate report failed")
        else:
            print("allure generate report sucess")
        os.environ["REPORT_SERVER_USERNAME"] = os.getenv("REPORT_SERVER_USERNAME")
        os.environ["REPORT_SERVER_PASSWORD"] = os.getenv("REPORT_SERVER_PASSWORD")
        os.environ["REPORT_SERVER"] = os.getenv("REPORT_SERVER")
        job_build_id = os.getenv("AGILE_JOB_BUILD_ID")
        REPORT_SERVER = os.getenv("REPORT_SERVER")
       
        cmd = "curl -s {}/report/upload.sh | bash -s ./report {} report".format(
                REPORT_SERVER, job_build_id)
        
        if job_build_id:
            # upload allure report
            cmd = "curl -s {}/report/upload.sh | bash -s ./report {} report".format(
                REPORT_SERVER, job_build_id)
            print("upload cmd is {}".format(cmd))
            ret = os.system(cmd)
        else:
            print("非流水线任务，请补充9位数字流水线任务id")
        
        if os.path.exists("allure-2.19.0.zip"):
            time.sleep(1)
            try:
                os.remove("allure-2.19.0.zip")
            except:
                print("#### can not remove allure-2.19.0.zip")
        if os.path.exists("java_linux.tar.gz"):
            time.sleep(1)
            try:
                os.remove("java_linux.tar.gz")
            except:
                print("#### can not remove java_linux.tar.gz")
        if os.path.exists("bos_new.tar.gz"):
            time.sleep(1)
            try:
                os.remove("bos_new.tar.gz")
            except:
                print("#### can not remove bos_new.tar.gz")
        return ret
    else:
        print("allure is not config correctly:{}, please config allure manually!".format(output))
        return 1


if __name__ == "__main__":
    gen_allure_report()