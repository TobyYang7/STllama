import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from util import *
from joblib import Parallel, delayed
import json
import pickle

DEFAULT_STHIS_TOKEN = "<ST_HIS>"
DEFAULT_STPRE_TOKEN = "<ST_PRE>"
DEFAULT_ST_START_TOKEN = "<ST_start>"
DEFAULT_ST_END_TOKEN = "<ST_end>"
DEFAULT_ST_PATCH_TOKEN = "<ST_patch>"

site_codes = ["4178000", "4182000", "4183000", "4183500", "4184500", "4185000", "4185318", "4185440", "4186500", "4188100",
              "4188496", "4189000", "4190000", "4191058", "4191444", "4191500", "4192500", "4192574", "4192599", "4193500"]
site_labels = {code: f"site {chr(i + ord('A'))}" for i, code in enumerate(site_codes)}


def batch_get_sediment_quantity(date_range, site_codes):
    sediment_data = pd.read_csv('discharge/SSC_sediment.csv', parse_dates=['date']).fillna(0)
    discharge_data = pd.read_csv('discharge/SSC_discharge.csv', parse_dates=['date'])

    sediment_data = sediment_data[sediment_data['date'].isin(date_range)]
    discharge_data = discharge_data[discharge_data['date'].isin(date_range)]

    sediment_data.set_index(['date'], inplace=True)
    discharge_data.set_index(['date'], inplace=True)

    return sediment_data, discharge_data


def process_site(site, date_range, sediment_data, discharge_data):
    N = len(date_range)
    data = np.zeros((N, 2))
    for j, date in enumerate(date_range):
        if date in sediment_data.index and str(site) in sediment_data.columns:
            data[j, 0] = sediment_data.loc[date, str(site)]
        if date in discharge_data.index and str(site) in discharge_data.columns:
            data[j, 1] = discharge_data.loc[date, str(site)]
    return data


def generate_spatio_temporal_data(start_date, end_date, site_codes):
    # 生成日期范围
    start_date = datetime.strptime(start_date, '%Y-%m-%d')
    end_date = datetime.strptime(end_date, '%Y-%m-%d')
    date_range = pd.date_range(start_date, end_date, freq='D')

    # 读取所有日期范围内的沉积物和水量数据
    sediment_data, discharge_data = batch_get_sediment_quantity(date_range, site_codes)

    # 并行处理每个站点的数据
    results = Parallel(n_jobs=-1)(delayed(process_site)(site, date_range, sediment_data, discharge_data) for site in site_codes)

    # 合并结果
    data = np.stack(results, axis=0)

    return data, date_range


def load_flow_direction_data():
    file_path = 'discharge/SSC_sites_flow_direction.csv'
    return pd.read_csv(file_path)


def get_direction(df, src, dst):
    src_str = str(src).zfill(8)
    dst_str = str(dst).zfill(8)
    try:
        direction = df.loc[df.iloc[:, 0] == int(src_str), dst_str].values[0]
    except KeyError:
        direction = None
    return direction


def generate_flow_direction_dict(site_codes, site_labels, flow_direction_data):
    flow_direction_dict = {}
    for site in site_codes:
        incoming_flows = []
        outgoing_flows = []

        for src in site_codes:
            if src != site:
                direction = get_direction(flow_direction_data, src, site)
                if direction is not None and direction > 0:
                    incoming_flows.append(f"from {site_labels[src]}")

        for dst in site_codes:
            if dst != site:
                direction = get_direction(flow_direction_data, site, dst)
                if direction is not None and direction > 0:
                    outgoing_flows.append(f"to {site_labels[dst]}")

        incoming_description = f"The water flows into this site: {', '.join(incoming_flows)}. " if incoming_flows else ""
        outgoing_description = f"The water flows out from this site: {', '.join(outgoing_flows)}. " if outgoing_flows else ""
        flow_description = incoming_description + outgoing_description

        flow_direction_dict[site] = flow_description

    return flow_direction_dict


def create_dialogue_json_and_st_data(data, date_range, site_codes, site_labels, flow_direction_dict):
    N, T, F = data.shape
    all_dialogues = []
    st_data_list = []

    idx_len = 0
    for t in range(T - 24):  # Ensure we have enough data for future 12 steps
        for n in range(N):
            sediment_values = data[n, t:t+12, 0]
            discharge_values = data[n, t:t+12, 1]
            future_sediment_values = data[n, t+12:t+24, 0]
            future_discharge_values = data[n, t+12:t+24, 1]
            start_date = date_range[t]
            end_date = date_range[t+11]
            site_label = site_labels[site_codes[n]]
            direction_info = flow_direction_dict[site_codes[n]]

            sediment_str = " ".join([f"{val:.2f}" for val in sediment_values])
            discharge_str = " ".join([f"{val:.2f}" for val in discharge_values])

            human_text = (
                f"Given the historical data for sediment and discharge over 12 time steps in {site_label}, "
                f"the recorded sediment values are [{sediment_str}], and the recorded discharge values are [{discharge_str}]. "
                f"The recording time of the historical data is '{start_date.strftime('%B %d, %Y, %H:%M, %A')} to {end_date.strftime('%B %d, %Y, %H:%M, %A')}, with data points recorded at 1-day intervals'. "
                f"Note: A sediment value of 0.00 indicates missing data. "
                f"Now we aim to predict the sediment and discharge in this region within the next 12 time steps. "
                f"{direction_info} To improve prediction accuracy, a spatio-temporal model is utilized to encode the historical data as tokens {DEFAULT_STHIS_TOKEN}, "
                f"where the first token corresponds to sediment and the second token corresponds to discharge. "
                f"Please conduct an analysis of the patterns in this region, considering the provided time and information, and then generate the prediction tokens for classification, in the form {DEFAULT_STPRE_TOKEN}."
            )
            gpt_text = f"Based on the given information, the predicted tokens in this region are {DEFAULT_STPRE_TOKEN}."

            dialogue_id = f"train_discharge_site_{n}_{n+1}_len_{idx_len // 20}"
            all_dialogues.append({
                "id": dialogue_id,
                "conversations": [{
                    "from": "human",
                    "value": human_text
                }, {
                    "from": "gpt",
                    "value": gpt_text
                }]
            })

            # 将数据填充到st_data_x和st_data_y
            st_data_x = np.zeros((1, 12, N, F))
            st_data_y = np.zeros((1, 12, N, F))
            st_data_x[0, :, n, 0] = sediment_values
            st_data_x[0, :, n, 1] = discharge_values
            st_data_y[0, :, n, 0] = future_sediment_values
            st_data_y[0, :, n, 1] = future_discharge_values

            if idx_len % 20 == 0:
                st_data_list.append({
                    "data_x": np.zeros((1, 12, N, F)),
                    "data_y": np.zeros((1, 12, N, F))
                })

            st_data_list[idx_len // 20]["data_x"][0, :, n, :] = st_data_x[0, :, n, :]
            st_data_list[idx_len // 20]["data_y"][0, :, n, :] = st_data_y[0, :, n, :]

            idx_len += 1

    return all_dialogues, st_data_list


if __name__ == "__main__":
    start_date = '2015-04-15'
    end_date = '2022-12-24'
    site_codes = ["4178000", "4182000", "4183000", "4183500", "4184500", "4185000", "4185318", "4185440", "4186500", "4188100",
                  "4188496", "4189000", "4190000", "4191058", "4191444", "4191500", "4192500", "4192574", "4192599", "4193500"]

    # 生成时空数据
    spatio_temporal_data, date_range = generate_spatio_temporal_data(start_date, end_date, site_codes)

    # 加载流向数据
    flow_direction_data = load_flow_direction_data()

    # 生成流向字典
    flow_direction_dict = generate_flow_direction_dict(site_codes, site_labels, flow_direction_data)

    # 生成对话数据集和st data
    dialogues, st_data_list = create_dialogue_json_and_st_data(spatio_temporal_data, date_range, site_codes, site_labels, flow_direction_dict)

    # 保存对话数据集为JSON文件
    with open('data/discharge/st_gpt.json', 'w') as f:
        json.dump(dialogues, f, indent=0)

    # 保存st data为pkl文件
    with open('data/discharge/st_data.pkl', 'wb') as f:
        pickle.dump(st_data_list, f)

    print(f"Dialogues and ST data generated successfully!")
