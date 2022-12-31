sed -i "s/AUTHOR_EMAIL/$(git config user.email)/" settings.ini
sed -i "s/AUTHOR/$(git config user.name)/" settings.ini
sed -i "s/USER/$(git config user.name)/" settings.ini
sed -i "s/LIB_NAME/$(pwd | awk -F/ '{print $NF}')/" settings.ini
