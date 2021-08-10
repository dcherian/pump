import pdb
import warnings
from xml.dom import minidom

import podaac as po
import podaac.podaac as podaac
import podaac.podaac_utils as utils
import tqdm
from podaac import drive as drive


def filter_night(string):
    return "MODIS_A-N-" in string


def make_wget_str(url, name):
    """chooses netCDF4 and subsets to needed variables."""
    return f"{url[:-5]}.nc4?lat,lon,time,sea_surface_temperature,quality_level -O 'datasets/modis/{name}'"


# make sure you encode your details in podaac.ini or passed explicitly, an example can be seen at https://podaacpy.readthedocs.io/en/latest/drive.html#drive.Drive
d = drive.Drive(None, username="dcherian", password="VSs59XrC4GSRvg58uDG")
p = podaac.Podaac()
u = utils.PodaacUtils()

# get from here: https://podaac.jpl.nasa.gov/ws/search/granule/index.html - OUT OF DATE
# or here: https://podaac.jpl.nasa.gov/dataset/MODIS_A-JPL-L2P-v2014.0?ids=Platform:ProcessingLevel&values=Aqua:*2*

kwargs = dict(
    dataset_id="PODAAC-GHMDA-2PJ02",
    start_time="2006-01-01T00:00:00Z",
    end_time="2010-12-31T00:00:00Z",
    bbox="-180,-10,-90,10",
    items_per_page="400",
)

# short_name = "JPL-L2P-MODIS_A"
# variables = p.dataset_variables(dataset_id)

# result = p.granule_preview(
#     dataset_id=dataset_id, image_variable="sea_surface_temperature"
# )

# result = p.granule_search(
#     dataset_id="PODAAC-ASOP2-25X01",
#     bbox="-75,30,-25,60",
#     start_time="2013-01-01T01:30:00Z",
#     end_time="2014-01-01T00:00:00Z",
#     start_index="1",
# )

# if bbox is specified, need the T...Z in start_time, end_time
result = p.granule_search(**kwargs, start_index="1")
doc = minidom.parseString(result)
num_granules = int(
    doc.getElementsByTagName("opensearch:totalResults")[0].firstChild.nodeValue
)

print(f"Found {num_granules} granules.")

nitems = int(kwargs["items_per_page"])
name_list = []
url_list = []
for start in range(1, num_granules + 1, nitems):
    for attempt in range(1, 11):
        print(f"starting index = {start} | attempt = {attempt}")
        result = p.granule_search(**kwargs, start_index=str(start))

        # this is useful: https://podaac.jpl.nasa.gov/forum/viewtopic.php?f=5&t=964
        names = u.mine_granules_from_granule_search(result)
        urls = u.mine_opendap_urls_from_granule_search(result)
        # granules = d.mine_drive_urls_from_granule_search(granule_search_response=result)
        # d.download_granules(granule_collection=granules, path=".")

        if start + len(names) != num_granules:
            try:
                assert len(names) == nitems
                assert len(urls) == nitems
            except AssertionError:
                print(
                    f"\n{len(names)} < {nitems} items returned. retrying attempt {attempt}..."
                )
            else:
                break
        else:
            break
    else:
        warnings.warn("Invalid data returned. even after 10 attempts.", UserWarning)
        pdb.set_trace()

    names = list(filter(filter_night, names))
    urls = list(filter(filter_night, urls))

    name_list += names
    url_list += urls


with open("url-list.txt", "w") as f:
    f.write("\n".join(map(make_wget_str, sorted(url_list), sorted(name_list))))
