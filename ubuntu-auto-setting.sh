# Ubuntu 20.04 auto setting script

# open firewall port
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw allow 3306/tcp

# apt update
sudo apt update

# Intall curl
sudo apt-get install curl

# Intall git
sudo apt-get install git

# Intall nodejs
curl -sL https://deb.nodesource.com/setup_18.x | sudo -E bash -
sudo apt-get install nodejs

# Install MySQL
sudo apt install mysql-server

mysql -u root -p

CREATE DATABASE `ea`

# import sql file
mysql -u root -p ea < Dump20230116.sql

ALTER USER 'root'@'localhost' IDENTIFIED BY 'password'; 
ALTER USER 'root'@'localhost' IDENTIFIED WITH mysql_native_password BY 'password';
