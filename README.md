Update the `ROOT_PASSWORD` and `PUBLIC_KEY` environment variables in the `docker-compose.yml`.

Deploy the CGAL environment with the following command.

```
docker compose -f docker-compose.yml up -d
```

The container's `sshd port 22` is mapped to the `host port 2200`.

The container's `xrdp port 3390` is mapped to the `host port 33890`.
