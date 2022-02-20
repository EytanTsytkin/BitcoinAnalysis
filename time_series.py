#bc1qq8empkvj3awkc5qjw7ve4k4qzs77kvyxhe2pvs.csv
#1LfYcbCsssB2niF3VWRBTVZFExzsweyPGQ.csv
#32ppaS6g3bZu5Qmz7Z9316aGG5YQaw41BL.csv
#1MwdgAgBHXqpQjahpxXAPxAXnuNQB8Gaoc.csv
#"1NGzsUxDoSXYtT4PDj4Yphi91ZaNTLTNU3.csv"

#####################################################################################
import pandas as pd
from datetime import datetime
from time import mktime
some_csv = "/root/address_vectors_merged/1MwdgAgBHXqpQjahpxXAPxAXnuNQB8Gaoc.csv"

wallet_df = pd.read_csv(some_csv)

# unix to date datetime.fromtimestamp
#

def str_time_to_unix(str_date):
    return mktime(datetime.strptime(str_date, TIME_CONV).timetuple())


def prepare_df(df):
    aggrigate_dict = {"valueBTC": "sum", "valueUSD": "sum", "feeBTC": "sum", "feeUSD": "sum", "time": "first"}
    df = df.groupby(["tx_index", "tx_type", "hash"],as_index=False).agg(aggrigate_dict)
    df["time"][0] = datetime.fromtimestamp(float(df["time"][0])).strftime('%Y-%m-%d %H:%M:%S')
    df["time_stamp"] = df.time.apply(str_time_to_unix)
    df.set_index("time", inplace=True)
    return df


