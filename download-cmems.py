#!/usr/bin/env python

# -*- coding: utf-8 -*-
#
# Script to loop on timespan (day / week / month or year) to optimize dataset requests (heavy in terms of number of files to be manipulated).


import os
import platform
import subprocess
import datetime as dt
import time
import calendar

# Copernicus Marine API Key - Login Credentials
# To create an account reach: http://marine.copernicus.eu/services-portfolio/register-now/.
# If already created but forgotten reach: http://marine.copernicus.eu/faq/forgotten-password/?idpage=169
username_cmems = "dcherian"
password_cmems = "CherianCMEMS2017%"

# Output directory name to store the Copernicus Marine data - (do not use whitespace character)
# If only 'folder-name' is given (not in absolute path), then it will be converted automatically into '$HOME/folder-name/'
local_storage_directory_name = "/home/deepak/pump/glorys"


# if string, writes commands to file
# if None, executes now
write_motu_commands_to_file = local_storage_directory_name + "/motu-commands.txt"

# - - - - - - - - - - - - - - - - - - - - - - - - -
# Product(s), Dataset(s) and MOTU server Parameters
# - - - - - - - - - - - - - - - - - - - - - - - - -

# CMEMS Variables &amp; Dataset ID &amp; Service ID &amp; MOTU server ID
# Define a dict to get required parameters of our daily temperature data request.
# It should looks like:
#                      {file_name (defined by yourself for practical reason): \
#                        [variable (aka -v), \
#                        dataset_id (aka -d), \
#                        product_id (aka -s), \
#                        motu_id (aka -m)]
#                        }

#  -v VARIABLE
#  --variable=VARIABLE
#                        The variable name or standard_name (list of strings, e.g. --variable=thetao or -v sea_water_potential_temperature)
#  -d PRODUCT_ID
#  --product-id=PRODUCT_ID
#                        The product (data set) to download (string e.g. -d global-analysis-forecast-phy-001-024)
#  -s SERVICE_ID
#  --service-id=SERVICE_ID
#                        The service identifier (string e.g. --service=GLOBAL_ANALYSIS_FORECAST_PHY_001_024-TDS)
#  -m MOTU
#  --motu=MOTU
#                        The motu server to use (url, e.g. -m http://nrt.cmems-du.eu/motu-web/Motu or --motu http://my.cmems-du.eu/motu-web/Motu)

# /!\ all CMEMS products are NOT hosted by a single server - they are grouped by MultiYear or NearRealTime products respectively on http://my.cmems-du.eu/motu-web/Motu and http://nrt.cmems-du.eu/motu-web/Motu
# You can always rely on the "VIEW SCRIPT" button of the Copernicus Marine Website (marine.copernicus.eu),
# using its DataExtraction WebInterface (also called GUI which stands for Graphical User Interface).
# It will generate the parameters of your extraction settings based on your selection.
# Please refer to this article to understand how to call/trigger this webservice/feature: http://marine.copernicus.eu/faq/how-to-write-and-run-the-script-to-download-cmems-products-through-subset-or-direct-download-mechanisms/?idpage=169

variables = ["thetao", "uo", "vo", "zos", "so"]

dataset_id = "global-reanalysis-phy-001-030-daily"
service_id = "GLOBAL_REANALYSIS_PHY_001_030-TDS"

# If True, one request per variable else all variables in one file
separate_variables = False

dict_id = {}  # mapping from file to request
if separate_variables:
    for var in variables:
        dict_id["{}".format(var)] = [
            "-v {}".format(var),
            "-d {}".format(dataset_id),
            "-s {}".format(service_id),
            "-m http://my.cmems-du.eu/motu-web/Motu",
        ]

else:
    dict_id["extract"] = [
        "-v thetao -v uo -v vo -v zos -v so",
        "-d {}".format(dataset_id),
        "-s {}".format(service_id),
        "-m http://my.cmems-du.eu/motu-web/Motu",
    ]

xmin_longitude = "-171"
xmax_longitude = "-94"
ymin_latitude = "-13"
ymax_latitude = "13"
zmin_depth = "0.493"
zmax_depth = "8000"

# Date - Timerange
yyyystart = 1997
mmstart = 1
yyyyend = 2000
mmend = 1
hhstart = " 12:00:00"
hhend = " 12:00:00"
dd = 1

delta_t = dt.timedelta(days=365 * 2)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# General Parameters - Tools - Proxy Network - Output Directory
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# Module declaration to the motu-client opensource-TOOLS to connect to MOTU CopernicusMarineHub.
# If you can't call it as module, then input the 'motu-client.py' absolute path. By default, usually in "Downloads" dir. after having followed the article on "python basic requirements":
# http://marine.copernicus.eu/faq/what-are-the-motu-and-python-requirements/?idpage=169
# Deprecated : motu_cl = '{absolute_path_to}/motu-client-python/motu-client.py'
# motu_cl = 'python -m motu-client'
motu_cl = ""

# File to log unsuccessful data extraction request(s)
logfile = "logfile.txt"

# Proxy Configuration
# Please replace "False" by "True" if you use a proxy to connect to internet and fill in the below variables.
proxy_flag = False
proxy_server_url = "http://your_proxy_url.com"
proxy_server_port = "port"
proxy_user_login = "your_proxy_user_login"
proxy_user_password = "your_proxy_user_password"


# Output prefix file name
pre_name = ""


# Check if output directory is well formated and if it exists, otherwise create it
absolute_path_substring = ["/home/", "C:\\"]
if local_storage_directory_name[-1] != "/":
    local_storage_directory_name = local_storage_directory_name + "/"
if not any(x in local_storage_directory_name for x in absolute_path_substring):
    local_storage_directory_name = (
        os.path.expanduser("~") + "/" + local_storage_directory_name
    )
if not os.path.exists(local_storage_directory_name):
    os.makedirs(local_storage_directory_name)

print("----------- Saving to {} -------------".format(local_storage_directory_name))

# Flags to let the server clears the buffer - better to be respectful when retrieving OPEN data
buffer_flag = False
cmd_flag = False

# Error Handle on dates (to illustrate an if statement with eval param '&gt;')
if yyyystart > yyyyend:
    print("[ERROR] in [Date Parameters]")
    print(
        """Please double check your date parameters, specifically the "yyyystart" which is currently greater than "yyyyend."""
    )
    print("""End of data extraction service.""")
    raise SystemExit

# Other variable definitions to be compatible with deprecated script versions still available on the Internet
log_cmems = "-u " + username_cmems
pwd_cmems = "-p " + password_cmems
pre_fic_cmd = "-f " + pre_name
out_cmd = "-o " + local_storage_directory_name
proxy_user = "--proxy-user " + proxy_user_login
proxy_pwd = "--proxy-pwd " + proxy_user_password
proxy_server = "--proxy-server " + proxy_server_url + ":" + proxy_server_port

zmin = "-z " + zmin_depth
zmax = "-Z " + zmax_depth

# To illustrate a simple Error Handle to delete a file when desired
try:
    os.remove(out_cmd.split()[1] + logfile)
except OSError:
    print("")

#print(
#    "\n+----------------------------+\n| ! - CONNEXION TO CMEMS HUB |\n+----------------------------+\n\n"
#)

if write_motu_commands_to_file:
    with open(write_motu_commands_to_file, "w") as file:
        file.write("")


boundaries = {
    "east": [xmax_longitude, xmax_longitude, ymin_latitude, ymax_latitude],
    "west": [xmin_longitude, xmin_longitude, ymin_latitude, ymax_latitude],
    "north": [xmin_longitude, xmax_longitude, ymax_latitude, ymax_latitude],
    "south": [xmin_longitude, xmax_longitude, ymin_latitude, ymin_latitude],
}
# To illustrate a For_Loop in order to generate download requests for several datasets held in a product
for key, value in dict_id.items():

    if buffer_flag:
        print(
            "Little pause to let the server clearing the buffer, it will AUTOMATICALLY resume once it's completed.\nNot mandatory but server-friendly :-)\n"
        )
        time.sleep(2)
        buffer_flag = False

    # Date declaration
    date_start = dt.datetime(yyyystart, mmstart, dd, 0, 0)
    date_end = dt.datetime(yyyyend, mmend, dd, 0, 0)

    # To illustrate a While_Loop in order to extract dailymean data, packed by month (Jan., Fev., Mar. etc...),
    # for as many download requests as number of months available in the timerange.
    while date_start <= date_end:
        for bound, (x0, x1, y0, y1) in boundaries.items():

            xmin = "-x " + x0
            xmax = "-X " + x1
            ymin = "-y " + y0
            ymax = "-Y " + y1

            timer_start = dt.datetime.now()
            # date_end_cmd = dt.datetime(date_start.year,
            #                           date_start.month,
            #                           calendar.monthrange(date_start.year, date_start.month)[1])

            date_end_cmd = date_start + delta_t
            if date_end_cmd > date_end:
                date_end_cmd = date_end

            date_cmd = (
                ' -t "'
                + date_start.strftime("%Y-%m-%d")
                + hhstart
                + '"'
                + ' -T "'
                + date_end_cmd.strftime("%Y-%m-%d")
                + hhend
                + '"'
            )

            fic_cmd = (
                pre_fic_cmd
                + key
                + "_"
                + bound
                + "_"
                + date_start.strftime("%Y-%m")
                + "_"
                + date_end_cmd.strftime("%Y-%m")
                + ".nc"
            )
            ficout = (
                pre_name
                + key
                + "_"
                + bound
                + "_"
                + date_start.strftime("%Y-%m")
                + "_"
                + date_end_cmd.strftime("%Y-%m")
                + ".nc"
            )

            print(
                "----------------------------------\n- ! - Processing dataset request : %s"
                % ficout
            )
            print("----------------------------------\n")
            if not os.path.exists(out_cmd.split()[1] + ficout):
                if not zmin_depth:
                    cmd = " ".join(
                        [
                            motu_cl,
                            log_cmems,
                            pwd_cmems,
                            value[3],
                            value[2],
                            value[1],
                            xmin,
                            xmax,
                            ymin,
                            ymax,
                            date_cmd,
                            value[0],
                            out_cmd,
                            fic_cmd,
                        ]
                    )
                else:
                    cmd = " ".join(
                        [
                            motu_cl,
                            log_cmems,
                            pwd_cmems,
                            value[3],
                            value[2],
                            value[1],
                            xmin,
                            xmax,
                            ymin,
                            ymax,
                            zmin,
                            zmax,
                            date_cmd,
                            value[0],
                            out_cmd,
                            fic_cmd,
                        ]
                    )
            if write_motu_commands_to_file:
                with open(write_motu_commands_to_file, "a") as file:
                    file.write(cmd + "\n")

            else:
                print("## MOTU API COMMAND ##")
                print(cmd)
                print(
                    "\n[INFO] CMEMS server is checking both your credentials and command syntax. If successful, it will extract the data and create your dataset on the fly. Please wait. \n"
                )
                subpro = subprocess.Popen(
                    cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
                )
                message, erreur = subpro.communicate()
                stat = subpro.returncode
                if stat != 0:
                    print("-- ERROR Incorrect Credentials :\n %s" % message)
                    with open(out_cmd.split()[1] + logfile, "a") as mylog:
                        mylog.write("Error : %s NOK\nDue to : %s" % (ficout, message))
                    print("""[INFO] Failed data extraction has been logged.\n""")
                    if b"HTTP Error 400" in message:
                        print(
                            """[INFO] Copernicus Marine USERNAME ('username_cmems') and/or PASSWORD ('password_cmems') are incorrect.\n\n[INFO] To execute the MOTU API COMMAND from your shell/terminal, please note the following rules:\n
                            On *nix OS, you must use the single quote, otherwise it may expand special characters.
                            [...] -u 'string' or --user='string' [...]\n
                            On Windows OS, you must use the double quote, because single quotes are treated literally.
                            [...] -p "string" or --pwd="string" [...]\n"""
                        )
                        raise SystemExit
                    if b"HTTP Error 407" in message:
                        print(
                            """[INFO] Proxy Authentication Required to connect to the Central Authentication System https://cmems-cas.cls.fr/cas/login\n\n[INFO] Check the value of proxy_flag (it should be True).\n\n[INFO] Double check your proxy settings:\n  --proxy-server=PROXY_SERVER\n                        the proxy server (url)\n  --proxy-user=PROXY_USER\n                        the proxy user (string)\n  --proxy-pwd=PROXY_PWD\n                        the proxy password (string)\n\n[INFO] If your proxy credentials are correct but your proxy password (string) contains a '@' then replace it by '%%40' """
                        )
                        print(
                            """[INFO] This issue is raised due either a misconfiguration in proxy settings or a network issue. If it persists, please contact your network administrator."""
                        )
                        raise SystemExit
                    if b"HTTP Error 403" in message:
                        print(
                            """[INFO] Copernicus Marine USERNAME ('username_cmems') has been suspended.\n[INFO] Please contact our Support Team either:\n  - By mail: servicedesk.cmems@mercator-ocean.eu or \n  - By using a webform, reaching the marine.copernicus.eu website and triggering the ANY QUESTIONS? button."""
                        )
                        raise SystemExit
                else:
                    if b"[ERROR]" in message:
                        print("-- ERROR Downloading command :\n %s" % message)
                        with open(out_cmd.split()[1] + logfile, "a") as mylog:
                            mylog.write(
                                "Error : %s NOK\nDue to : %s" % (ficout, message)
                            )
                        print("""[INFO] Failed data extraction has been logged.\n""")
                    else:
                        print(
                            "-- MOTU Download successful :\n %s OK\n"
                            % fic_cmd.split()[1]
                        )
                        cmd_flag = True
        else:
            print(
                "-- Your dataset for %s has already been downloaded in %s --\n"
                % (fic_cmd.split()[1], out_cmd.split()[1])
            )
            cmd_flag = False

        date_start = date_end_cmd + dt.timedelta(days=1)
        print(
            "------------- Timer: {} ------------".format(
                dt.datetime.now() - timer_start
            )
        )

    if cmd_flag:
        buffer_flag = True
        cmd_flag = False

if not os.path.exists(out_cmd.split()[1] + logfile):
    print(
        "\n------------------------------------------------\n - ! - Your Copernicus Dataset(s) are located in %s\n------------------------------------------------\n"
        % (out_cmd.split()[1])
    )
else:
    print("## [ERROR] ##")
    print(
        "/!\\ Some download requests failed. Please see recommendation in %s%s"
        % (out_cmd.split()[1], logfile)
    )
#print(
#    "+--------------------------------------------+\n| ! - CONNEXION TO CMEMS HUB HAS BEEN CLOSED |\n+--------------------------------------------+\n"
#)
