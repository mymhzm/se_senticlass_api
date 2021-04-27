from flask_wtf import FlaskForm
from wtforms import TextAreaField, SubmitField, StringField
from wtforms.validators import DataRequired

class comment_submit(FlaskForm):
    content = TextAreaField(validators=[DataRequired()])
    submit = SubmitField('Review')

class user_login(FlaskForm):
    username = StringField(validators=[DataRequired()])
    password = StringField(validators=[DataRequired()])
    submit = SubmitField('Login')