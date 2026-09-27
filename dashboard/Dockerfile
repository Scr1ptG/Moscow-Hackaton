FROM nginx:1.27-alpine
COPY dashboard/nginx.conf /etc/nginx/conf.d/default.conf
COPY dashboard/index.html dashboard/app.js dashboard/style.css /usr/share/nginx/html/
EXPOSE 80
