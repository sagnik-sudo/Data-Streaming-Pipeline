import streamlit as st
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader

# Configuration for the authenticator
config = {
    'credentials': {
        'usernames': {
            'user1': {
                'name': 'User One',
                'password': 'password1'
            },
            'user2': {
                'name': 'User Two',
                'password': 'password2'
            }
        }
    },
    'cookie': {
        'expiry_days': 30,
        'key': 'some_signature_key',
        'name': 'some_cookie_name'
    }
}

# Save the configuration to a YAML file
with open('config.yaml', 'w') as file:
    yaml.dump(config, file, default_flow_style=False)

# Load the configuration from the YAML file
with open('config.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)

# Create the authenticator
authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days']
)

# User login
name, authentication_status, username = authenticator.login('Login', 'main')

if authentication_status:
    st.write(f'Welcome *{name}*')
    st.title('Streamlit App')
    st.write('This is a protected Streamlit app.')
    authenticator.logout('Logout', 'main')
elif authentication_status == False:
    st.error('Username/password is incorrect')
elif authentication_status == None:
    st.warning('Please enter your username and password')

# User registration (optional)
if st.button('Register'):
    try:
        authenticator.register_user('Register user', preauthorized=True)
    except Exception as e:
        st.error(e)

# Run the Streamlit app
if __name__ == "__main__":
    st.run()