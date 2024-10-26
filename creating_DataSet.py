from pandas.core.interchange.dataframe_protocol import DataFrame
from getting_DataFrame import *

def change_the_data(typev: int,value):
    if type(value) == str:
        value = value.replace(' ', '')
    match value:
        case 'Микробизнес':
            return 0
        case 'Малыйбизнес':
            return 1
        case 'Среднийбизнес':
            return 2
        case 'Крупныйбизнес':
            return 3
        case None:
            return 0
        case '':
            return 0
        case _:
            if typev == 4:
                return 1
            else:
                return value

def marketing_list(id: int, df: DataFrame):
    res = []
    rows = [df.iloc[i].to_list() for i in range(len(df['ID']))]
    for row in rows:
        if row[0] == id:
            for i in range(len(row)):
                row[i] = change_the_data(i, row[i])
            res = row
            break
    if len(res) > 0:
        cols = ['ID', 'company_size', 'capital_size', 'emp_amount', 'is_els', 'payment_index','risk_index']
    else:
        cols = []
    return res,cols

def requests(id: int, df: DataFrame):
    rows = [df.iloc[i].to_list() for i in range(len(df['ID']))]
    reports = 0
    for row in rows:
        if row[0] == id and row[1] == 'Жалобы':
            reports += 1
    l = [reports]
    return l, ['reports_amount']

def interests(id: int, df: DataFrame):
    res_dict = {
        'Завершен неудачно': 0,
        'Завершен успешно': 0,
        'Регистрация клиента на "РЖД Маркет"': 0,
        'Отказ в работе': 0,
        'Раз. предложения\Офор.заказа на "РЖД Маркет"': 0
    }
    rows = [df.iloc[i] for i in range(len(df['ID']))]
    for row in rows:
        if row.iloc[1] == id:
            if row.iloc[0] in res_dict.keys():
                res_dict[row.iloc[0]] += 1
    return list(res_dict.values()), ['fail', 'success', 'registration', 'denied', 'once_offer']

def target(id: int, df: DataFrame):
    rows = [df.iloc[i].to_list() for i in range(len(df['ID']))]
    targ = 1
    ischoosed = False
    for row in rows:
        if row[0] == id:
            targ = row[1]
            ischoosed = True
        elif ischoosed:
            break
    if ischoosed:
        return [targ], ['target']
    else:
        return [0], ['target']

def create_dataset(paths: list, output: str, is_for_train: bool):
    id_table = pd.read_excel(paths[-1])
    tables = list()
    for i in range(8):
        tables.append(pd.read_excel(paths[i]).drop(columns=['Находится в реестре МСП', 'ОКВЭД2.Наименование', 'ОКВЭД2.Код','Город фактический',
                                       'Город юридический','Грузоотправитель','Грузополучатель',
                                       'Карточка клиента (внешний источник).Индекс платежной дисциплины Описание',
                                       'Карточка клиента (внешний источник).Индекс финансового риска Описание',
                                       'Госконтракты.Контракт','Госконтракты.Тип контракта'], errors='ignore').fillna(0))
    tables.append(pd.read_excel(paths[8]).drop(columns=['Дата','Тема','Сценарий','Подразделение','Ожидаемая выручка','Вероятность сделки, %','Дата следующей активности',
                                          'Следующая активность','Канал первичного интереса','Номер',
                                           'Ссылка (служебное поле для вывода на экран прочих реквизитов объекта)'], errors='ignore'))
    tables.append(pd.read_excel(paths[9]).drop(columns=['Дата','Тема','Номер','Тип обращения','Группа вопросов','Количество доработок'], errors='ignore'))
    tables.append(pd.read_excel(paths[10], skiprows=2).drop(columns=['Субъект федерации отп','Субъект федерации наз',
                                                       'Код груза','Гр груза по опер.номен'], errors='ignore'))
    if is_for_train:
        tables.append(pd.read_excel(paths[11]))
    ids = [i for i in id_table['ID']]
    predataset = list()
    columns1 = list()
    columns = list()
    end = False

    for id in range(len(ids)):
        temp = []
        end = False
        for i in range(len(tables)):
            if i < 8 and not end:
                temp1, ctemp = marketing_list(ids[id], tables[i])
                if i == 0:
                    temp += temp1
                    print(temp1)
                elif len(temp1) > 0:
                    temp = temp1
                    print(temp1)
                if not 'ID' in columns1:
                    columns1 += ctemp
                if len(temp) > 0 and temp[1] != 0:
                    end = True

            elif i == 8:
                temp1, ctemp = interests(ids[id], tables[i])
                temp += temp1
                if not ctemp[0] in columns1:
                    columns1 += ctemp
            elif i == 9:
                temp1, ctemp = requests(ids[id], tables[i])
                temp += temp1
                if not ctemp[0] in columns1:
                    columns1 += ctemp
            elif is_for_train and i == 11:
                temp1, ctemp = target(ids[id], tables[i])
                temp += temp1
                if not ctemp[0] in columns1:
                    columns1 += ctemp
        predataset.append(temp)
        columns = columns1
    for data in range(len(predataset)):
        assert len(predataset[data]) == len(columns), f'id: {data} | {len(predataset[data]), predataset[data], len(columns), columns}'
    try:
        dataset = pd.DataFrame(data = predataset, columns = columns)
    except ValueError:
        print(len(columns), len(predataset[0]))
        print(columns)
        print(predataset)
        dataset = pd.DataFrame(data = [], columns = [])
    dataset.to_csv(output)