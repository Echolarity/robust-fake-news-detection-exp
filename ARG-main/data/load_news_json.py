import json


def load_json(file_path):
    count = 0
    fake_count = 0
    real_count = 0
    data_list = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                line_data = json.loads(line)
                count += 1
                if line_data.get("label") == 1:
                    real_count += 1
                elif line_data.get("label") == 0:
                    fake_count += 1
                data_list.append(line_data)
            except Exception as e:
                print(f"An error occurred: {e}")
    real_rate = real_count / count
    return data_list, real_rate, count, real_count, fake_count
