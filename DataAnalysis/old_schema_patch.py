import sqlite3

#旧データベーススキーマの場合の変更処理
# データベースに接続
db_name = 'sk326Gen120min.db'
conn = sqlite3.connect(db_name)
cursor = conn.cursor()

try:
# 新しいテーブルを作成（すべてのカラムをコピーし、変更したいカラム名だけ変更）
    cursor.execute('''
    CREATE TABLE cells_new AS 
    SELECT 
        id, 
        cell_id,
        label_experiment,
        manual_label,
        perimeter,
        area,
        img_ph,
        img_fluo AS img_fluo1,
        contour,
        center_x,
        center_y
    FROM cells
    ''')
except:
    pass

# 旧テーブルを削除
try:
    cursor.execute('DROP TABLE cells')
except:
    pass
# 新テーブルの名前を旧テーブルの名前に変更
try:
    cursor.execute('ALTER TABLE cells_new RENAME TO cells')
except:
    pass

# コミットと接続のクローズ
conn.commit()
conn.close()
