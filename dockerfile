# Docker file for running gatsby without installing node version 10 or Gatsby.
# Hayley Boyce, February 6th, 2020

FROM node:10

# Add the package.json file and build the node_modules folder
WORKDIR /app
COPY ./package*.json ./
RUN mkdir node_modules && npm install
RUN npm install --global gatsby-cli && gatsby telemetry --disable