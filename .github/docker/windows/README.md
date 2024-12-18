# docker-github-runner-windows

Docker codes for building a self hosted GitHub runner as a windows container.

Typically, one would build the docker image using the `docker-compose.yml` file with:

```
docker compose build --build-arg RUNNER_VERSION=2.318.0
```
Check latest releases at [GitHub Actions Runner](https://github.com/actions/runner/releases).

To run the container, the `docker-compose.yml` file needs a `variable.env` file
containing these variables:

```
GH_OWNER=FelipeCybis
GH_REPOSITORY=physbeh
GH_TOKEN=<your-github-pat-token>
```

Now, the container can be run with:

```
docker compose up --scale runner=1 -d
```

(or `runner=x` for running `x` containers).

For more details on using it, check out this blog post: [Self Hosted GitHub Runners on Azure - Windows Container](https://dev.to/pwd9000/create-a-docker-based-self-hosted-github-runner-windows-container-3p7e).

Also checkout this GitHub repository from where the base image comes from:
[docker-github-runner-windows](https://github.com/Pwd9000-ML/docker-github-runner-windows).  
Or a similar one for Linux-based containers: [docker-github-runner-linux](https://github.com/Pwd9000-ML/docker-github-runner-linux).
