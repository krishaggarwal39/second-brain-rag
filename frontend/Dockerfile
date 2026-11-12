# Stage 1: Build the React application
FROM node:26.13.0-alpine AS build

WORKDIR /app
COPY package.json package-lock.json ./
# Reproducible install from the lockfile.
RUN npm ci
COPY . .
# The app talks to /api; nginx reverse-proxies it to the backend container.
RUN echo "VITE_API_URL=/api" > .env.production
RUN npm run build

# Stage 2: Serve with an unprivileged nginx (runs as uid 101, listens on 8080)
FROM nginxinc/nginx-unprivileged:1.27.3-alpine

COPY --from=build /app/dist /usr/share/nginx/html

# Template rendered by the stock entrypoint. Only the variables listed in
# NGINX_ENVSUBST_FILTER are substituted, so nginx runtime variables ($host, ...)
# are never touched.
COPY nginx.conf /etc/nginx/templates/default.conf.template
COPY --chmod=0755 docker/05-require-api-key.sh /docker-entrypoint.d/05-require-api-key.sh

ENV NGINX_ENVSUBST_FILTER="^(API_KEY|REAL_IP_FROM)$" \
    REAL_IP_FROM=127.0.0.1/32

EXPOSE 8080

CMD ["nginx", "-g", "daemon off;"]
