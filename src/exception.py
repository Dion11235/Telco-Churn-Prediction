import sys
from src.logger import logging

class CustomException(Exception):
    def __init__(self, error_detail):
        super().__init__(str(error_detail))
        self.custom_error_msg = self.custom_error_detail(str(error_detail), error_detail)
        logging.info(self.custom_error_msg)
    
    def custom_error_detail(self, error_message, error_detail):
        file_name = error_detail.__traceback__.tb_frame.f_code.co_filename
        line_no = error_detail.__traceback__.tb_lineno
        return f"Error occurred in - {file_name} - at line - {line_no} ; error message : {error_message}"
    
    def __str__(self):
        return self.custom_error_msg
    

if __name__ == "__main__":
    try:
        a = 1 / 0
    except Exception as e:
        logging.info("Dummy testing -- "+str(e))
        raise CustomException(e)
