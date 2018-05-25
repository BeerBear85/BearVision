import logging, os, csv
import gpx_parser

logger = logging.getLogger(__name__)  #Set logger to reflect the current file

output_file_ending = "_bear_vision_GPS_format"
# The BearVision format (csv):
# time [YYYYMMDD_HH_mm_SS_FF], latitude [deg], longitude [deg], accuracy [m], speed [m/s], number of satellites [-]

class InputGPS_Importer:
    def __init__(self):
        return

    # Should parse all the found different types of GPS files in the input folder and generate GPS files in the BearVision format
    @staticmethod
    def import_user_format_gps_files(arg_input_folder: str, arg_output_folder_path: str):

        for gps_file in os.scandir(arg_input_folder):
            output_path = InputGPS_Importer.__create_output_path(gps_file, arg_output_folder_path)
            return_val = True
            if gps_file.name.endswith('.gpx') and gps_file.is_file():
                return_val = InputGPS_Importer.__generate_from_gpx_file(gps_file, output_path)
            elif gps_file.name.endswith('.tcx') and gps_file.is_file():
                return_val = InputGPS_Importer.__generate_from_tcx_file(gps_file, output_path)

        return

    @staticmethod
    def __generate_from_gpx_file(arg_input_file: os.DirEntry, arg_output_path: str):
        logger.info("Generating GPS output file: " + arg_output_path + " from input file: " + arg_input_file.path)
        gpx_file_object = open(arg_input_file.path, 'r')
        gpx_data = gpx_parser.parse(gpx_file_object)

        with open(arg_output_path, 'w', newline='') as output_file:
            output_writer = csv.writer(output_file)
            for track in gpx_data.tracks:
                for segment in track.segments:
                    for point in segment.points:
                        output_writer.writerow([
                            point.time.strftime("%Y%m%d_%H_%M_%S_%f"),
                            point.latitude,
                            point.longitude,
                            point.speed,
                            point.horizontal_dilution,
                            point.satellites
                        ])

        return True


    @staticmethod
    def __generate_from_tcx_file(arg_input_file: os.DirEntry, arg_output_path: str):
        return False

    @staticmethod
    def __create_output_path(arg_input_file: os.DirEntry, arg_output_folder_path: str):
        if not (os.path.isdir(arg_output_folder_path) and os.path.exists(arg_output_folder_path)):
            raise ValueError("Output folder is not a valid folder: " + arg_output_folder_path)

        output_filename_short = os.path.splitext(arg_input_file.name)[0] + output_file_ending + ".csv"
        output_dir = os.path.dirname(arg_input_file.path)
        output_path = os.path.join(arg_output_folder_path, output_filename_short)
        return output_path
