# extract_config.py

import configparser

def config_dict(config_path):
    """
    INI 형식 설정 파일을 읽어들여
    섹션별로 타입이 적절히 변환된 dict를 반환합니다.
    """
    parser = configparser.ConfigParser()
    parser.read(config_path)

    cfg = {}
    for section in parser.sections():
        section_dict = {}
        for key, val in parser[section].items():
            # 정수 변환 시도
            try:
                section_dict[key] = parser.getint(section, key)
                continue
            except ValueError:
                pass
            # 실수 변환 시도
            try:
                section_dict[key] = parser.getfloat(section, key)
                continue
            except ValueError:
                pass
            # 불리언 변환 시도
            low = val.lower()
            if low in ('true', 'false'):
                section_dict[key] = parser.getboolean(section, key)
            else:
                section_dict[key] = val
        cfg[section] = section_dict

    return cfg
