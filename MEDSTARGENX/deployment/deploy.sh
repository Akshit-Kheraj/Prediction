#!/bin/bash

##############################################################################
# MEDSTARGENX Deployment Script for Ubuntu 24.04
# Domain: mgenx.com
# Server: 72.62.241.114 (srv1264932.hstgr.cloud)
##############################################################################

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}╔════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║                                                        ║${NC}"
echo -e "${GREEN}║     🚀 MEDSTARGENX Deployment Script                  ║${NC}"
echo -e "${GREEN}║        Ubuntu 24.04 VPS Deployment                    ║${NC}"
echo -e "${GREEN}║                                                        ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════╝${NC}"
echo ""

# Configuration
DOMAIN="mgenx.com"
APP_DIR="/var/www/medstargenx"
REPO_URL="https://github.com/Akshit-Kheraj/Prediction.git"
MONGODB_VERSION="7.0"

export DEBIAN_FRONTEND=noninteractive

# Step 1: System Update
echo -e "${YELLOW}[1/10] Updating system packages...${NC}"
sudo apt update
sudo apt upgrade -y

# Step 2: Install Node.js 20.x
echo -e "${YELLOW}[2/10] Installing Node.js 20.x...${NC}"
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt install -y nodejs
echo "Node version: $(node -v)"
echo "NPM version: $(npm -v)"

# Step 3: Install MongoDB
echo -e "${YELLOW}[3/10] Installing MongoDB ${MONGODB_VERSION}...${NC}"
sudo apt-get install -y gnupg curl

# Download key to file first to avoid TTY issues
curl -fsSL https://www.mongodb.org/static/pgp/server-${MONGODB_VERSION}.asc -o mongo.asc
sudo gpg --batch --yes --dearmor -o /usr/share/keyrings/mongodb-server-${MONGODB_VERSION}.gpg mongo.asc
rm -f mongo.asc

echo "deb [ arch=amd64,arm64 signed-by=/usr/share/keyrings/mongodb-server-${MONGODB_VERSION}.gpg ] https://repo.mongodb.org/apt/ubuntu jammy/mongodb-org/${MONGODB_VERSION} multiverse" | \
   sudo tee /etc/apt/sources.list.d/mongodb-org-${MONGODB_VERSION}.list
sudo apt update
sudo apt install -y mongodb-org
sudo systemctl start mongod
sudo systemctl enable mongod
echo "MongoDB status:"
sudo systemctl status mongod --no-pager

# Step 4: Install Python and dependencies
echo -e "${YELLOW}[4/10] Installing Python 3.12 and pip...${NC}"
sudo apt install -y python3.12 python3.12-venv python3-pip

# Step 5: Install Nginx
echo -e "${YELLOW}[5/10] Installing Nginx...${NC}"
sudo apt install -y nginx
sudo systemctl start nginx
sudo systemctl enable nginx

# Step 6: Install PM2 globally
echo -e "${YELLOW}[6/10] Installing PM2 process manager...${NC}"
sudo npm install -g pm2
pm2 startup systemd -u root --hp /root

# Step 7: Clone repository
echo -e "${YELLOW}[7/10] Cloning repository...${NC}"
sudo rm -rf $APP_DIR
sudo mkdir -p $APP_DIR
sudo git clone $REPO_URL $APP_DIR
cd $APP_DIR

# Step 8: Setup Frontend
echo -e "${YELLOW}[8/10] Setting up Frontend...${NC}"
npm install
npm run build
echo "Frontend built successfully!"

# Step 9: Setup Backend
echo -e "${YELLOW}[9/10] Setting up Backend...${NC}"
cd $APP_DIR/backend
npm install

# Create backend .env file
cat > .env << EOF
NODE_ENV=production
PORT=5000
MONGODB_URI=mongodb://localhost:27017/medstargenx
JWT_SECRET=$(openssl rand -base64 32)
JWT_EXPIRE=7d
JWT_REFRESH_SECRET=$(openssl rand -base64 32)
JWT_REFRESH_EXPIRE=30d
FRONTEND_URL=https://${DOMAIN}
RATE_LIMIT_WINDOW_MS=900000
RATE_LIMIT_MAX_REQUESTS=100
EOF

echo "Backend .env created!"

# Start backend with PM2
cd $APP_DIR
sudo mkdir -p /var/log/medstargenx
pm2 start deployment/ecosystem.config.js
pm2 save

# Step 10: Setup ML API
echo -e "${YELLOW}[10/10] Setting up ML API...${NC}"
cd $APP_DIR
python3.12 -m venv venv
source venv/bin/activate
pip install --upgrade pip
cd ml_api
pip install -r requirements.txt

# Copy systemd service
sudo cp $APP_DIR/deployment/ml_api.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl start ml_api
sudo systemctl enable ml_api

# Configure Nginx
echo -e "${YELLOW}Configuring Nginx...${NC}"
sudo cp $APP_DIR/deployment/nginx.conf /etc/nginx/sites-available/medstargenx
sudo ln -sf /etc/nginx/sites-available/medstargenx /etc/nginx/sites-enabled/
sudo rm -f /etc/nginx/sites-enabled/default
sudo nginx -t
sudo systemctl reload nginx

# Setup firewall
echo -e "${YELLOW}Configuring UFW firewall...${NC}"
sudo ufw allow 'Nginx Full'
sudo ufw allow OpenSSH
sudo ufw --force enable

echo ""
echo -e "${GREEN}╔════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║                                                        ║${NC}"
echo -e "${GREEN}║     ✅ DEPLOYMENT COMPLETED SUCCESSFULLY!              ║${NC}"
echo -e "${GREEN}║                                                        ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${GREEN}🌐 Frontend:${NC} http://${DOMAIN}"
echo -e "${GREEN}🔌 Backend API:${NC} http://${DOMAIN}/api"
echo -e "${GREEN}🤖 ML API:${NC} http://${DOMAIN}/ml/api"
echo ""
echo -e "${YELLOW}Service Status:${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
sudo systemctl status mongod --no-pager | head -5
echo ""
pm2 status
echo ""
sudo systemctl status ml_api --no-pager | head -5
echo ""
sudo systemctl status nginx --no-pager | head -5
echo ""
echo -e "${YELLOW}Next Steps:${NC}"
echo "1. Configure DNS: Point mgenx.com to 72.62.241.114"
echo "2. Setup SSL: Run 'sudo certbot --nginx -d mgenx.com -d www.mgenx.com'"
echo "3. Update Nginx config to enable HTTPS redirect"
echo ""
