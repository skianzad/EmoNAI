Source pack — build or run on another computer
==============================================

This folder has the Next.js app/ directory. It is not the production zip.

1) Install Node.js 20+

2) In this folder:

   npm install
   npm run dev

   Open http://localhost:3000
   Password: 637942

3) To make the server bundle (no app/ folder — that is normal):

   npm run pack:server

   Then copy deploy.zip (or the deploy folder) to the Linux box and:

   unzip deploy.zip -d ~/genai-app
   cd ~/genai-app
   pm2 start ecosystem.config.cjs
   pm2 save
