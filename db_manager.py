import re
import zipfile
from zipfile import ZipFile
import pandas as pd
from glob import glob
from tqdm import tqdm


class DBManager(object):
    """DBManager helps locating all the archives containing stock data from year 1990 to 2018"""
    
    def __init__(self, db_root_path, recursion_level=2, archive_extension='zip'):
        self.root_path = db_root_path
        self.recursion_level = recursion_level
        self.archive_ext = "*.%s" % archive_extension
        self.locate_archives()
        

    def locate_archives(self):
        """ Locate paths to all the archives"""
        path_pattern = self.root_path + "*/"*self.recursion_level + self.archive_ext
        self.archive_paths = sorted(glob(path_pattern))
        self._to_path_dict()


    def _to_path_dict(self):
        archive_names = [s.split('/')[-1][:-len(self.archive_ext)+1] for s in self.archive_paths]
        self.path_dict = {}

        try:
            assert len(archive_names) > 0
        except Exception as e:
            print("No data found in the database!")
            raise e

        try:
            pattern = re.compile("[A-Z.]+_\d{4}")
            for archive_name_str, archive_path in zip(archive_names, self.archive_paths):
                if pattern.match(archive_name_str):
                    symbol, year = archive_name_str.split('_')
                    if symbol not in self.path_dict:
                        self.path_dict[symbol] = {year: archive_path}
                    else:
                        self.path_dict[symbol][year] = archive_path
        except Exception as e:
            raise e

    def open_archive(self, symbol, year):
        try:
            if symbol in self.path_dict and str(year) in self.path_dict[symbol]:
                archive = ZipFile(self.path_dict[str(symbol)][str(year)])
                return archive
            else:
                return None
        except Exception as e:
            raise e

    def close_archive(self, archive):
        archive.close()


    def unzip_data(self, archive, data_name):
        with archive.open(data_name) as data:
            try:
                df = pd.read_csv(data)
                return df

            except pd.errors.EmptyDataError:
                print(data)
                print("Skip this file because it is empty")

    def get_unzipped_data(self, symbol, year, last_few_days = None):
        archive = self.open_archive(symbol, year)

        if archive is None or len(archive.namelist())<6: 
            print("ohhh {} {} zip file is empty or less than 6 days".format(symbol, year))
            return

        # Else, continue with actual work:
        if last_few_days is None:
            file_list = sorted(archive.namelist())
        else:
            file_list = sorted(archive.namelist())[-last_few_days:]
            # print("get last {} days data {}".format(last_few_days, file_list))

        data_list = []
        for data_name in tqdm(file_list, desc="Unzipping {} {}".format(symbol, year)):
            unzipped_data = self.unzip_data(archive, data_name)

            if unzipped_data is None: 
                continue
            else: 
                data_list.append(unzipped_data)
        # print("unzipping {} {} file completed".format(symbol, year))

        return data_list






