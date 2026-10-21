/**
 * PM2 config for the production standalone server.
 *
 * On the build machine (Windows): run `npm run pack:server` → zip.
 * On the server: unzip, then `pm2 start ecosystem.config.cjs`
 *
 * Proxy / CA are for the Huawei gateway box. Adjust paths if needed.
 */
const path = require("path");
const os = require("os");

const home = process.env.HOME || process.env.USERPROFILE || os.homedir();
const caCert =
  process.env.NODE_EXTRA_CA_CERTS ||
  path.join(home, "HuaweiWebSecureInternetGatewayCAV2_soheil.pem");

module.exports = {
  apps: [
    {
      name: "webapp-genai",
      cwd: __dirname,
      script: "server.js",
      instances: 1,
      exec_mode: "fork",
      autorestart: true,
      max_memory_restart: "1G",
      env: {
        NODE_ENV: "production",
        PORT: 3000,
        HOSTNAME: "0.0.0.0",
        HTTP_PROXY: process.env.HTTP_PROXY || "http://127.0.0.1:3146",
        HTTPS_PROXY: process.env.HTTPS_PROXY || "http://127.0.0.1:3146",
        NODE_USE_ENV_PROXY: "1",
        NODE_EXTRA_CA_CERTS: caCert,
      },
    },
  ],
};
