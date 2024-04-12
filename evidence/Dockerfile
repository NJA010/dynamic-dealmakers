FROM node:20.10-bookworm as builder

WORKDIR /reports
COPY package-lock.json .
COPY package.json .

RUN npm ci

COPY . .
RUN sed -i  's/localhost/host.docker.internal/' sources/dd/connection.yaml

ARG EVIDENCE_SOURCE__dynamic_dealmakers__connectionString=${EVIDENCE_SOURCE__dynamic_dealmakers__connectionString}
ENV EVIDENCE_SOURCE__dynamic_dealmakers__connectionString=${EVIDENCE_SOURCE__dynamic_dealmakers__connectionString}
RUN npm run sources && npm run build

FROM nginx:1.25.3-bookworm

COPY --chmod=777 --from=builder /reports/build /usr/share/nginx/html
