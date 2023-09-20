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
,                           total_data, mutihop sensor_data,    uav_data
uav_greedy (感測器資料量大的),                     9108,                   4759,           4349
greedy_and_mutihop,         9108,       4257        1804,           2597
drift_greedy_and_mutihop,   9108,       4257        1804,           2597
Q_learning,                 9108,       4257        1674,           2834
mutihop(OK),                    9108,       4257,       0

爆掉的資料量

封包抵達率
UAV代收的資料量 
overflow 

時間 1000 單位

至少六張圖

感測器的資料經過計算後 回傳成功 (感測器的資料經過計算後 回傳失敗 (斷路且飛機找不到))
感測器的資料來不及計算導致 overflow (飛機找不到 飛機來不及到)

節點飄移距離參數改變
無人機電池電量 
感測器 buffer 大小改變

2 x 3 

找現成的感測器 buffer 大小
