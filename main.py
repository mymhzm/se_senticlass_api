from flask import Flask, request, render_template, redirect, make_response, flash, url_for
from forms import comment_submit, user_login
from emotion_model.prediction_api_demo import lstm_classifier
import sqlite3
import os
import sys
dir_path = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__)
app.config['DEBUG'] = True
app.config['SECRET_KEY'] = 'pa55w0rd'

#
# ---------------------------------------------------------------------------------------------------------
# classifier model initialization
lstm_action = lstm_classifier()
lstm_action.data_init()
#
# # input one sentence and return a emotion result
def emotion_classifier(inputs):
    res_emotion = lstm_action.predict(inputs)[0]
    return res_emotion

# ---------------------------------------------------------------------------------------------------------

database = os.path.join(dir_path, "database/comment.db")

# db actions
def create_connection(db_file):
    conn = None
    try:
        conn = sqlite3.connect(db_file)
    except:
        print("Connection fails")

    return conn

def creat_comment_table():
    conn = create_connection(database)
    cursor = conn.cursor()
    cursor.execute('''
                CREATE TABLE IF NOT EXISTS Comment_Table(
                    cid INTEGER PRIMARY KEY autoincrement,
                    comment text, 
                    Timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )''')
    conn.commit()

def creat_user_table():
    conn = create_connection(database)
    cursor = conn.cursor()
    cursor.execute('''
                CREATE TABLE IF NOT EXISTS User(
                    uid INTEGER PRIMARY KEY autoincrement,
                    username text, 
                    password text,
                    gp text
            )''')
    conn.commit()

def insert_comment_into_db(comment):
    conn = create_connection(database)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO Comment_Table (comment) VALUES (?)", (comment,))
    conn.commit()

def extract_data_from_db():
    conn = create_connection(database)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM Comment_Table ORDER BY Timestamp DESC")

    rows = cursor.fetchall()
    comment_list = []

    for row in rows:
        r = {}
        r['id'] = row[0]
        r['comment'] = row[1]
        r['time'] = row[2]
        comment_list.append(r)

    return comment_list

# authentication
def sign_in(username, password):
    conn = create_connection(database)
    cursor = conn.cursor()
    find_user = ("SELECT * FROM User WHERE username = ? AND password = ?")
    cursor.execute(find_user, [(username), (password)])
    result = cursor.fetchall()

    return result

# flask route
@app.route("/")
def home():
    creat_comment_table()
    creat_user_table()
    return render_template("index.html")

@app.route("/product", methods=['GET'])
def product():
    creat_comment_table()
    
    comment_form = comment_submit()
    login_form = user_login()
    
    return render_template("product_page.html", form=comment_form,  login_form=login_form, comment_list=extract_data_from_db())

# submit comment
@app.route("/comment", methods=['POST'])
def comment():
    comment_form = comment_submit()
    login_form = user_login()

    if comment_form.is_submitted():
        content = comment_form.content.data
        insert_comment_into_db(content)
        return redirect(url_for('product'))
    
    return render_template("product_page.html", form=comment_form,  login_form=login_form, comment_list=extract_data_from_db())

# user login
@app.route("/login", methods=['POST'])
def login():
    comment_form = comment_submit()
    login_form = user_login()

    if login_form.is_submitted():
        username = login_form.username.data
        password = login_form.password.data
        result = sign_in(username, password)

        if result:
            resp = make_response(redirect('/product'))
            resp.set_cookie('userID', str(result[0][0]))
            resp.set_cookie('userName', str(result[0][1]))
            resp.set_cookie('userGroup', str(result[0][3]))

            return resp

        else:
            flash('Wrong user name or password')
            return redirect(url_for('product'))

    return render_template('product_page.html', form=comment_form, login_form=login_form, comment_list=extract_data_from_db())


@app.route("/admin", methods=['GET'])
def admin():
    creat_comment_table()
    e = {'anger': 12, 'fear': 2,'joy': 25,'love': 42,'sadness': 6,'surprise': 10}

    return render_template("admin_portal.html", comment_list=extract_data_from_db(), emotions=e)

if __name__ == '__main__':
   app.run()
