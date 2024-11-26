def convert_rows_to_transaction_details(rows: list) -> list:
    transactions = []
    for row in rows:
        transaction = {
            "Time": row[0],
            "V1": row[1],
            "V2": row[2],
            "V3": row[3],
            "V4": row[4],
            "V5": row[5],
            "V6": row[6],
            "V7": row[7],
            "V8": row[8],
            "V9": row[9],
            "V10": row[10],
            "V11": row[11],
            "V12": row[12],
            "V13": row[13],
            "V14": row[14],
            "V15": row[15],
            "V16": row[16],
            "V17": row[17],
            "V18": row[18],
            "V19": row[19],
            "V20": row[20],
            "V21": row[21],
            "V22": row[22],
            "V23": row[23],
            "V24": row[24],
            "V25": row[25],
            "V26": row[26],
            "V27": row[27],
            "V28": row[28],
            "Amount": row[29]
        }
        transactions.append(transaction)
    return transactions