class SchemaProvider:
    """Provides schema details for the database."""

    @staticmethod
    def get_nba_schema_2() -> str:
        return """
        0|Team|TEXT eg. "Toronto Raptors"
        1|NAME|TEXT eg. "Otto Porter Jr."
        2|Jersey|TEXT eg. "0" and when null has a value "NA"
        3|POS|TEXT eg. "PF"
        4|AGE|INT eg. "22" in years
        5|HT|TEXT eg. `6' 7"` or `6' 10"`
        6|WT|TEXT eg. "232 lbs"
        7|COLLEGE|TEXT eg. "Michigan" and when null has a value "--"
        8|SALARY|TEXT eg. "$9,945,830" and when null has a value "--"
        """

    @staticmethod
    def get_nba_schema_1() -> str:
        return """\
        0|Team|TEXT 
        1|NAME|TEXT  
        2|Jersey|TEXT 
        3|POS|TEXT
        4|AGE|INT 
        5|HT|TEXT 
        6|WT|TEXT 
        7|COLLEGE|TEXT 
        8|SALARY|TEXT
        """
