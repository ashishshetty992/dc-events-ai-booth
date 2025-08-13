# Optimized Frontend Dockerfile for Railway
FROM node:18-alpine AS builder

WORKDIR /app

# Copy package files
COPY package*.json ./

# Install dependencies
RUN npm ci --only=production

# Copy source code
COPY . .

# Build the application for production
RUN npm run build

# Production stage with nginx
FROM nginx:alpine

# Copy built assets
COPY --from=builder /app/dist /usr/share/nginx/html

# Create custom nginx config for Railway
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Railway assigns PORT dynamically
EXPOSE $PORT

# Create startup script that uses Railway's PORT
RUN echo '#!/bin/sh' > /start.sh && \
    echo 'sed -i "s/80/$PORT/g" /etc/nginx/conf.d/default.conf' >> /start.sh && \
    echo 'nginx -g "daemon off;"' >> /start.sh && \
    chmod +x /start.sh

CMD ["/start.sh"]
