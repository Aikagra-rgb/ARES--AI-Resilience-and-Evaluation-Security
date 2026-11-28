FROM mcr.microsoft.com/dotnet/sdk:8.0 AS build
WORKDIR /src

COPY Ares.Web.csproj ./
RUN dotnet restore Ares.Web.csproj

COPY . ./
RUN dotnet publish Ares.Web.csproj -c Release -o /app/publish /p:UseAppHost=false

FROM mcr.microsoft.com/dotnet/aspnet:8.0 AS runtime
WORKDIR /app

RUN apt-get update && apt-get upgrade -y && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /var/lib/ares/keys && chown -R app:app /var/lib/ares/keys

COPY --from=build /app/publish ./

USER app
ENV ASPNETCORE_URLS=http://+:8080
EXPOSE 8080

HEALTHCHECK --interval=15s --timeout=5s --start-period=20s --retries=3 \
    CMD ["bash", "-c", "exec 3<>/dev/tcp/127.0.0.1/8080"]

ENTRYPOINT ["dotnet", "Ares.Web.dll"]
