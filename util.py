import pandas as pd
import matplotlib.pyplot as plt

site_codes = ["4178000", "4182000", "4183000", "4183500", "4184500", "4185000", "4185318", "4185440", "4186500", "4188100",
              "4188496", "4189000", "4190000", "4191058", "4191444", "4191500", "4192500", "4192574", "4192599", "4193500"]
site_labels = {code: f"site {chr(i + ord('A'))}" for i, code in enumerate(site_codes)}


def get_direction(src, dst):
    """
    从 SSC_sites_flow_direction.csv 中获取流向数据
    输入: src (源站点), dst (目标站点)
    输出: direction (流向)
    """
    file_path = 'discharge/SSC_sites_flow_direction.csv'
    df = pd.read_csv(file_path)
    src_str = str(src).zfill(8)
    dst_str = str(dst).zfill(8)
    try:
        direction = df.loc[df.iloc[:, 0] == int(src_str), dst_str].values[0]
    except KeyError:
        direction = None

    return direction


def get_sediment(date, site):
    """
    从 SSC_sediment.csv 中获取沉积物数据
    输入: date (日期), site (站点)
    输出: sediment (沉积物浓度)
    """
    try:
        sediment_data = pd.read_csv('discharge/SSC_sediment.csv', parse_dates=['date'])
        sediment_data = sediment_data.fillna(0)  # 将缺失值替换为0
        sediment_value = sediment_data.loc[sediment_data['date'] == date, str(site)].values[0]
        return sediment_value
    except (KeyError, IndexError):
        return None


def get_quantity(date, site):
    """
    从 SSC_discharge.csv 中获取水量数据
    输入: date (日期), site (站点)
    输出: quantity (水量)
    """
    try:
        discharge_data = pd.read_csv('discharge/SSC_discharge.csv', parse_dates=['date'])
        quantity_value = discharge_data.loc[discharge_data['date'] == date, str(site)].values[0]
        return quantity_value
    except (KeyError, IndexError):
        return None


def plot_data(start_date, end_date, site):
    """
    绘制指定时间范围内某站点的沉积物变化趋势图，标记缺失值的时间戳
    输入: start_date (起始时间), end_date (终止时间), site (站点)
    """
    # 读取沉积物数据
    sediment_data = pd.read_csv('discharge/SSC_sediment.csv', parse_dates=['date'])

    # 筛选指定时间范围的数据
    mask = (sediment_data['date'] >= start_date) & (sediment_data['date'] <= end_date)
    filtered_data = sediment_data.loc[mask]

    # 获取指定站点的沉积物数据
    site_column = str(site)
    if site_column not in filtered_data.columns:
        print(f"Site {site} not found in the data.")
        return

    sediment_values = filtered_data[site_column]
    missing_dates = filtered_data['date'][sediment_values.isna()]

    # 绘制图表
    plt.figure(figsize=(20, 6))
    plt.plot(filtered_data['date'], sediment_values, linestyle='-', label='Sediment Concentration')
    plt.scatter(missing_dates, [0]*len(missing_dates), color='red', label='Missing Values', zorder=1, s=1)

    plt.xlabel('Date')
    plt.ylabel('Sediment')
    plt.title(f'Sediment at Site {site} from {start_date} to {end_date}')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
