library(tidyverse)
library(readxl)
library(janitor)
library(lubridate)
library(stringr)
library(xlsx) 
library(readr)
library(astsa)
library(dplyr)
library(lmtest)
library(naniar)
library(rugarch)
# Volumen de Credito ------------------------------------------------------

base_2005 <- read_excel("SB Volumen Credito/volumen_ene_dic_2005.xlsx",sheet = 1) %>% clean_names() %>% 
  mutate(region = NA) %>% rename(tipo_de_operacion = region) %>%
  add_column(estado_de_la_operacion = NA, .after = 5)
base_2006 <- read_excel("SB Volumen Credito/volumen_ene_dic_2006.xlsx",sheet = 1) %>% clean_names() %>% 
  mutate(region = NA) %>% rename(tipo_de_operacion = region) %>%
  add_column(estado_de_la_operacion = NA, .after = 5)
base_2007 <- read_excel("SB Volumen Credito/volumen_ene_dic_2007.xlsx",sheet = 1) %>% clean_names() %>% 
  mutate(region = NA) %>% rename(tipo_de_operacion = region) %>%
  add_column(estado_de_la_operacion = NA, .after = 5)
base_2008 <- read_excel("SB Volumen Credito/volumen_ene_dic_2008.xlsx",sheet = 1) %>% clean_names() %>% 
  mutate(region = NA) %>% rename(tipo_de_operacion = region) %>%
  add_column(estado_de_la_operacion = NA, .after = 5)
base_2009 <- read_excel("SB Volumen Credito/volumen_ene_dic_2009.xlsx",sheet = 1) %>% clean_names() %>% 
  mutate(region = NA) %>% rename(tipo_de_operacion = region) %>%
  add_column(estado_de_la_operacion = NA, .after = 5)
base_2010 <- read_excel("SB Volumen Credito/volumen_ene_dic_2010.xlsx",sheet = 1) %>% clean_names() %>% 
  mutate(region = NA) %>% rename(tipo_de_operacion = region) %>%
  add_column(estado_de_la_operacion = NA, .after = 5)
base_2011 <- read_excel("SB Volumen Credito/volumen_ene_dic_2011.xlsx",sheet = 1) %>% clean_names() %>% 
  mutate(region = NA) %>% rename(tipo_de_operacion = region) %>%
  add_column(estado_de_la_operacion = NA, .after = 5)
base_2012 <- read_excel("SB Volumen Credito/volumen_ene_dic_2012.xlsx",sheet = 1) %>% clean_names() %>%
  add_column(estado_de_la_operacion = NA, .after = 5)
base_2013 <- read_excel("SB Volumen Credito/volumen_ene_dic_2013.xlsx",sheet = 1) %>% clean_names() %>%
  add_column(estado_de_la_operacion = NA, .after = 5)
base_2014 <- read_excel("SB Volumen Credito/volumen_ene_dic_2014.xlsx",sheet = 1) %>% clean_names() %>%
  add_column(estado_de_la_operacion = NA, .after = 5)
base_2015 <- read_excel("SB Volumen Credito/volumen_ene_dic_2015.xlsx",sheet = 1) %>% clean_names() %>% 
  select(1:11,monto_otorgado, numero_de_operaciones)
base_2016 <- read_excel("SB Volumen Credito/volumen_ene_dic_2016.xlsx",sheet = 1) %>% clean_names() %>% 
  select(1:11,monto_otorgado, numero_de_operaciones)
base_2017 <- read_excel("SB Volumen Credito/volumen_ene_dic_2017.xlsx",sheet = 1) %>% clean_names() %>% 
  select(1:11,monto_otorgado, numero_de_operaciones)
base_2018 <- read_excel("SB Volumen Credito/volumen_ene_dic_2018.xlsx",sheet = 1) %>% clean_names() %>% 
  select(1:11,monto_otorgado, numero_de_operaciones)
base_2019 <- read_excel("SB Volumen Credito/volumen_ene_dic_2019.xlsx",sheet = 1) %>% clean_names() %>% 
  select(1:11,monto_otorgado, numero_de_operaciones)
base_2020 <- read_excel("SB Volumen Credito/volumen_ene_dic_2020.xlsx",sheet = 1) %>% clean_names() %>% 
  select(1:11,monto_otorgado, numero_de_operaciones)
base_2021 <- read_excel("SB Volumen Credito/volumen_ene_dic_2021.xlsx",sheet = 1) %>% clean_names() %>% 
  select(1:11,monto_otorgado, numero_de_operaciones)
base_2022 <- read_excel("SB Volumen Credito/volumen_ene_dic_2022_05.xlsx",sheet = 1) %>% clean_names() %>% 
  select(1:11,monto_otorgado, numero_de_operaciones)

base_cv <- rbind(base_2005,base_2006,base_2007,base_2008,base_2009,base_2010,
                 base_2011,base_2012,base_2013,base_2014,base_2015,base_2016,
                 base_2017,base_2018,base_2019,base_2020,base_2021,base_2022)
base_cv <- base_cv %>% 
  mutate(subsistema = str_replace_all(subsistema,
                                      c("BANCOS PRIVADOS NACIONALES"="BANCOS PRIVADOS",
                                        "BANCOS PRIVADOS EXTRANJEROS"="BANCOS PRIVADOS")),
         tipo_de_credito = str_squish(tipo_de_credito),
         tipo_de_operacion = str_squish(tipo_de_operacion),
         estado_de_la_operacion = str_squish(estado_de_la_operacion),
         fecha = date(fecha)) %>% 
  drop_na(tipo_de_credito)
  

base_cv$tipo_de_credito[base_cv$tipo_de_credito == "CONSUMO"] <- "CONSUMO"
base_cv$tipo_de_credito[base_cv$tipo_de_credito == "EDUCATIVO"] <- "CONSUMO"
base_cv$tipo_de_credito[base_cv$tipo_de_credito == "CONSUMO PRIORITARIO"] <- "CONSUMO"
base_cv$tipo_de_credito[base_cv$tipo_de_credito == "CONSUMO ORDINARIO"] <- "CONSUMO"
base_cv$tipo_de_credito[base_cv$tipo_de_credito == "EDUCATIVO DE INTERES SOCIAL"] <- "CONSUMO"
base_cv$tipo_de_credito[base_cv$tipo_de_credito == "EDUCATIVO SOCIAL"] <- "CONSUMO"
base_cv$tipo_de_credito[base_cv$tipo_de_credito == "#N/A"] <- "CONSUMO"
base_cv$tipo_de_credito[base_cv$tipo_de_credito == "VIVIENDA"] <- "HIPOTECARIO"
base_cv$tipo_de_credito[base_cv$tipo_de_credito == "INMOBILIARIO"] <- "HIPOTECARIO"
base_cv$tipo_de_credito[base_cv$tipo_de_credito == "VIVIENDA INTERES PUBLICO"] <- "HIPOTECARIO"
base_cv$tipo_de_credito[base_cv$tipo_de_credito == "VIVIENDA INTERES SOCIAL"] <- "HIPOTECARIO"
base_cv$tipo_de_credito[base_cv$tipo_de_credito == "COMERCIAL"] <- "PRODUCTIVO"
base_cv$tipo_de_credito[base_cv$tipo_de_credito == "COMERCIAL PRIORITARIO CORPORATIVO"] <- "PRODUCTIVO"
base_cv$tipo_de_credito[base_cv$tipo_de_credito == "COMERCIAL PRIORITARIO EMPRESARIAL"] <- "PRODUCTIVO"
base_cv$tipo_de_credito[base_cv$tipo_de_credito == "COMERCIAL PRIORITARIO PYMES"] <- "PRODUCTIVO"
base_cv$tipo_de_credito[base_cv$tipo_de_credito == "MICROCREDITO DE ACUMULACION SIMPLE"] <- "PRODUCTIVO"
base_cv$tipo_de_credito[base_cv$tipo_de_credito == "MICROCREDITO DE ACUMULACION AMPLIADA"] <- "PRODUCTIVO"
base_cv$tipo_de_credito[base_cv$tipo_de_credito == "MICROCREDITO MINORISTA"] <- "PRODUCTIVO"
base_cv$tipo_de_credito[base_cv$tipo_de_credito == "COMERCIAL ORDINARIO"] <- "PRODUCTIVO"
base_cv$tipo_de_credito[base_cv$tipo_de_credito == "PRODUCTIVO EMPRESARIAL"] <- "PRODUCTIVO"
base_cv$tipo_de_credito[base_cv$tipo_de_credito == "PRODUCTIVO PYMES"] <- "PRODUCTIVO"
base_cv$tipo_de_credito[base_cv$tipo_de_credito == "PRODUCTIVO CORPORATIVO"] <- "PRODUCTIVO"
base_cv$tipo_de_credito[base_cv$tipo_de_credito == "MICROCREDITO AGRICULTURA Y GANADERIA"] <- "PRODUCTIVO"
base_cv$tipo_de_credito[base_cv$tipo_de_credito == "PRODUCTIVO AGRICULTURA Y GANADERIA"] <- "PRODUCTIVO"
base_cv$tipo_de_credito[base_cv$tipo_de_credito == "MICROCRÉDITO AGRICULTURA Y GANADERÍA"] <- "PRODUCTIVO"
base_cv$tipo_de_credito[base_cv$tipo_de_credito == "PRODUCTIVO AGRICULTURA Y GANADERÍA"] <- "PRODUCTIVO"
base_cv$tipo_de_credito[base_cv$tipo_de_credito == "FACTORING"] <- "PRODUCTIVO"
base_cv$tipo_de_credito[base_cv$tipo_de_credito == "MICROCREDITO"] <- "PRODUCTIVO"
base_cv$tipo_de_credito[base_cv$tipo_de_credito == "COMERCIAL CORPORATIVO"] <- "PRODUCTIVO"
base_cv$tipo_de_credito[base_cv$tipo_de_credito == "COMERCIAL EMPRESARIAL"] <- "PRODUCTIVO"
base_cv$tipo_de_credito[base_cv$tipo_de_credito == "COMERCIAL PYMES"] <- "PRODUCTIVO"
base_cv$tipo_de_credito[base_cv$tipo_de_credito == "INVERSION PUBLICA"] <- "PRODUCTIVO"
base_cv$tipo_de_credito[is.na(base_cv$tipo_de_credito)] <- "PRODUCTIVO"
# Probar si el tipo de operacion CONTINGENTE solo se da cuando el credito es PRODUCTIVO

#Col 5
base_cv <- base_cv %>% 
  mutate(tipo_de_operacion = str_replace_all(tipo_de_operacion,c("#N/A"="CREDITO"))) %>% 
  replace_na(list(tipo_de_operacion = "CREDITO"))

#Col 6
base_cv <- base_cv %>% replace_na(list(estado_de_la_operacion = "ORIGINAL"))
#Col 12

#Col 13

# Convertir a factor las var char

base_cv$tipo_de_credito <- factor(base_cv$tipo_de_credito, levels = unique(base_cv$tipo_de_credito))
base_cv$tipo_de_operacion <- factor(base_cv$tipo_de_operacion, levels = unique(base_cv$tipo_de_operacion))
#base_cv$estado_de_la_operacion <- factor(base_cv$estado_de_la_operacion, levels = unique(base_cv$estado_de_la_operacion))

base_cvf <- base_cv %>% filter(subsistema == "BANCOS PRIVADOS" & tipo_de_operacion == "CREDITO") %>%
  select(1,4,12) %>% group_by(fecha, tipo_de_credito) %>% summarise(volumen_credito = sum(monto_otorgado)/1000000)

base_cvf <- pivot_wider(base_cvf, names_from = tipo_de_credito,values_from = volumen_credito) %>% 
  mutate(credito_total = PRODUCTIVO + CONSUMO + HIPOTECARIO, credito_hogares = CONSUMO + HIPOTECARIO) %>% 
  clean_names()

#base_cvf %>% as.data.frame() %>% write.xlsx("base_cvf.xlsx")

# Google Queries ----------------------------------------------------------

# Credito Hipotecario

credito_banco_guayaquil <- read_csv("Google Queries/credito_banco_guayaquil.csv", col_names =T,col_select = c(2)) %>% clean_names()
credito_banco_pacifico <- read_csv("Google Queries/credito_banco_pacifico.csv", col_names =T,col_select = c(2)) %>% clean_names()
credito_banco_pichincha <- read_csv("Google Queries/credito_banco_pichincha.csv", col_names =T,col_select = c(2)) %>% clean_names()
credito_hipotecario <- read_csv("Google Queries/credito_hipotecario.csv", col_names =T,col_select = c(2)) %>% clean_names()
prestamo_hipotecario <- read_csv("Google Queries/prestamo_hipotecario.csv", col_names =T,col_select = c(2)) %>% clean_names()
simulador_de_credito <- read_csv("Google Queries/simulador_de_credito.csv", col_names =T,col_select = c(2)) %>% clean_names()
credito <- read_csv("Google Queries/credito.csv", col_names =T,col_select = c(2)) %>% clean_names()
prestamo <- read_csv("Google Queries/prestamo.csv", col_names =T,col_select = c(2)) %>% clean_names()
query_hipotecario <- cbind(credito_banco_guayaquil, credito_banco_pacifico, credito_banco_pichincha,
                           credito_hipotecario, prestamo_hipotecario, simulador_de_credito, credito, prestamo)
# Credito Consumo
credito_banco_pichincha <- read_csv("Google Queries/credito_banco_pichincha.csv", col_names =T,col_select = c(2)) %>% clean_names()
simulador_de_credito <- read_csv("Google Queries/simulador_de_credito.csv", col_names =T,col_select = c(2)) %>% clean_names()
credito_produbanco <- read_csv("Google Queries/credito_produbanco.csv", col_names =T,col_select = c(2)) %>% clean_names()
credito_quirografario <- read_csv("Google Queries/credito_quirografario.csv", col_names =T,col_select = c(2)) %>% clean_names()
prestamo_quirografario <- read_csv("Google Queries/prestamo_quirografario.csv", col_names =T,col_select = c(2)) %>% clean_names()
credito_banco_guayaquil <- read_csv("Google Queries/credito_banco_guayaquil.csv", col_names =T,col_select = c(2)) %>% clean_names()
credito <- read_csv("Google Queries/credito.csv", col_names =T,col_select = c(2)) %>% clean_names()
prestamo <- read_csv("Google Queries/prestamo.csv", col_names =T,col_select = c(2)) %>% clean_names()
query_consumo <- cbind(credito_banco_pichincha, simulador_de_credito, credito_produbanco, credito_quirografario,
                           prestamo_quirografario, credito_banco_guayaquil, credito, prestamo)

# Credito Hogares

credito_banco_guayaquil <- read_csv("Google Queries/credito_banco_guayaquil.csv", col_names =T,col_select = c(2)) %>% clean_names()
credito_banco_pacifico <- read_csv("Google Queries/credito_banco_pacifico.csv", col_names =T,col_select = c(2)) %>% clean_names()
credito_banco_pichincha <- read_csv("Google Queries/credito_banco_pichincha.csv", col_names =T,col_select = c(2)) %>% clean_names()
simulador_de_credito <- read_csv("Google Queries/simulador_de_credito.csv", col_names =T,col_select = c(2)) %>% clean_names()
credito_quirografario <- read_csv("Google Queries/credito_quirografario.csv", col_names =T,col_select = c(2)) %>% clean_names()
prestamo_quirografario <- read_csv("Google Queries/prestamo_quirografario.csv", col_names =T,col_select = c(2)) %>% clean_names()
credito <- read_csv("Google Queries/credito.csv", col_names =T,col_select = c(2)) %>% clean_names()
prestamo <- read_csv("Google Queries/prestamo.csv", col_names =T,col_select = c(2)) %>% clean_names()
credito_produbanco <- read_csv("Google Queries/credito_produbanco.csv", col_names =T,col_select = c(2)) %>% clean_names()

query_hogar <- cbind(credito_banco_guayaquil, credito_banco_pacifico, credito_banco_pichincha, simulador_de_credito,
                     credito_quirografario, prestamo_quirografario, credito,prestamo,credito_produbanco) %>% 
  replace_with_na_all(condition = ~. ==0)
  

# Macroeconomic Indicators ------------------------------------------------

ind_mac <- read_excel("Indicadores macro/ind_macro.xlsx", sheet = 1, range = cell_cols(2:10)) %>%
  clean_names() %>% select(-c(3,5))


# Base Final --------------------------------------------------------------

base_hipotecario <- cbind(base_cvf %>% select(1,4), query_hipotecario, ind_mac)
base_consumo <- cbind(base_cvf %>% select(1,3), query_consumo,ind_mac) %>% as.data.frame()
base_hogares <- cbind(base_cvf %>% select(1,6), query_hogar, ind_mac) %>% as.data.frame()

# Feature Selection -------------------------------------------------------------

# acf(base_hipotecario[,"hipotecario"])
# ccf(as.numeric(unlist(base_hipotecario[37:208,"iaec"])) %>% diff(), as.numeric(unlist(base_hipotecario[37:208,"hipotecario"])) %>% diff(), lag.max = 30)
# #lag2.plot(as.numeric(unlist(base_hipotecario[37:208,"hipotecario"])), as.numeric(unlist(base_hipotecario[37:208,"iaec"])), 10)
# ccf(as.numeric(unlist(base_hipotecario[37:208,"icc"])) %>% diff(), as.numeric(unlist(base_hipotecario[37:182,"hipotecario"])) %>% diff(), lag.max = 30)
# ccf(as.numeric(unlist(base_hipotecario[37:207,"ice"])) %>% diff(), as.numeric(unlist(base_hipotecario[37:207,"hipotecario"])) %>% diff(), lag.max = 30)
# ccf(as.numeric(unlist(base_hipotecario[37:208,"inflacion"])) %>% diff(), as.numeric(unlist(base_hipotecario[37:208,"hipotecario"])) %>% diff(), lag.max = 30)
# ccf(as.numeric(unlist(base_hipotecario[37:208,"m2"])) %>% diff(), as.numeric(unlist(base_hipotecario[37:208,"hipotecario"])) %>% diff(), lag.max = 30)
# ccf(as.numeric(unlist(base_hipotecario[37:208,"roe_sf"])) %>% diff(), as.numeric(unlist(base_hipotecario[37:208,"hipotecario"])) %>% diff(), lag.max = 30)
# ccf(as.numeric(unlist(base_hipotecario[37:208,"precio_wti"])) %>% diff(), as.numeric(unlist(base_hipotecario[37:208,"hipotecario"])) %>% diff(), lag.max = 30)
# ccf(as.numeric(unlist(base_hipotecario[37:208,"tasa_activa"])) %>% diff(), as.numeric(unlist(base_hipotecario[37:208,"hipotecario"])) %>% diff(), lag.max = 30)
# ccf(as.numeric(unlist(base_hipotecario[37:208,"tasa_pasiva"])) %>% diff(), as.numeric(unlist(base_hipotecario[37:208,"hipotecario"])) %>% diff(), lag.max = 30)


## Correlacion cruzada sin tendencia
# acf(as.numeric(unlist(base_consumo[,"consumo"])) %>% diff())
# ccf(as.numeric(unlist(base_consumo[37:208,"credito_banco_pichincha_ecuador"])) %>% diff(), as.numeric(unlist(base_consumo[37:208,"consumo"])) %>% diff(), lag.max = 30)
# ccf(as.numeric(unlist(base_consumo[37:208,"simulador_de_credito_ecuador"])) %>% diff(), as.numeric(unlist(base_consumo[37:208,"consumo"])) %>% diff(), lag.max = 30)
# ccf(as.numeric(unlist(base_consumo[37:208,"credito_produbanco_ecuador"])) %>% diff(), as.numeric(unlist(base_consumo[37:208,"consumo"])) %>% diff(), lag.max = 30)
# ccf(as.numeric(unlist(base_consumo[37:208,"credito_quirografario_ecuador"])) %>% diff(), as.numeric(unlist(base_consumo[37:208,"consumo"])) %>% diff(), lag.max = 30)
# ccf(as.numeric(unlist(base_consumo[37:208,"prestamo_quirografario_ecuador"])) %>% diff(), as.numeric(unlist(base_consumo[37:208,"consumo"])) %>% diff(), lag.max = 30)
# ccf(as.numeric(unlist(base_consumo[37:208,"credito_banco_guayaquil_ecuador"])) %>% diff(), as.numeric(unlist(base_consumo[37:208,"consumo"])) %>% diff(), lag.max = 30)
# ccf(as.numeric(unlist(base_consumo[37:208,"credito_ecuador"])) %>% diff(), as.numeric(unlist(base_consumo[37:208,"consumo"])) %>% diff(), lag.max = 30)
# ccf(as.numeric(unlist(base_consumo[37:208,"prestamo_ecuador"])) %>% diff(), as.numeric(unlist(base_consumo[37:208,"consumo"])) %>% diff(), lag.max = 30)
# 
# ccf(as.numeric(unlist(base_consumo[37:208,"iaec"])) %>% diff(), as.numeric(unlist(base_consumo[37:208,"consumo"])) %>% diff(), lag.max = 30)
# ccf(as.numeric(unlist(base_consumo[37:208,"icc"])) %>% diff(), as.numeric(unlist(base_consumo[37:182,"consumo"])) %>% diff(), lag.max = 30)
# ccf(as.numeric(unlist(base_consumo[37:207,"ice"])) %>% diff(), as.numeric(unlist(base_consumo[37:207,"consumo"])) %>% diff(), lag.max = 30)
# ccf(as.numeric(unlist(base_consumo[37:208,"inflacion"])) %>% diff(), as.numeric(unlist(base_consumo[37:208,"consumo"])) %>% diff(), lag.max = 30)
# ccf(as.numeric(unlist(base_consumo[37:208,"m2"])) %>% diff(), as.numeric(unlist(base_consumo[37:208,"consumo"])) %>% diff(), lag.max = 30)
# ccf(as.numeric(unlist(base_consumo[37:208,"roe_sf"])) %>% diff(), as.numeric(unlist(base_consumo[37:208,"consumo"])) %>% diff(), lag.max = 30)
# ccf(as.numeric(unlist(base_consumo[37:208,"precio_wti"])) %>% diff(), as.numeric(unlist(base_consumo[37:208,"consumo"])) %>% diff(), lag.max = 30)
# ccf(as.numeric(unlist(base_consumo[37:208,"tasa_activa"])) %>% diff(), as.numeric(unlist(base_consumo[37:208,"consumo"])) %>% diff(), lag.max = 30)
# ccf(as.numeric(unlist(base_consumo[37:208,"tasa_pasiva"])) %>% diff(), as.numeric(unlist(base_consumo[37:208,"consumo"])) %>% diff(), lag.max = 30)

## Correlacion cruzada con tendencia
# acf(base_consumo[,"consumo"])
# pacf(base_consumo[,"consumo"])
# ccf(base_consumo[37:208,"credito_banco_pichincha_ecuador"], base_consumo[37:208,"consumo"], lag.max = 30)
# ccf(base_consumo[37:208,"simulador_de_credito_ecuador"], base_consumo[37:208,"consumo"], lag.max = 30)
# ccf(base_consumo[37:208,"credito_produbanco_ecuador"], base_consumo[37:208,"consumo"], lag.max = 30)
# ccf(base_consumo[37:208,"credito_quirografario_ecuador"], base_consumo[37:208,"consumo"], lag.max = 30)
# ccf(base_consumo[37:208,"prestamo_quirografario_ecuador"], base_consumo[37:208,"consumo"], lag.max = 30)
# ccf(base_consumo[37:208,"credito_banco_guayaquil_ecuador"], base_consumo[37:208,"consumo"], lag.max = 30)
# ccf(base_consumo[37:208,"credito_ecuador"], base_consumo[37:208,"consumo"], lag.max = 30)
# ccf(base_consumo[37:208,"prestamo_ecuador"], base_consumo[37:208,"consumo"], lag.max = 30)
# 
# ccf(base_consumo[37:208,"iaec"], base_consumo[37:208,"consumo"], lag.max = 30)
# ccf(base_consumo[37:182,"icc"], base_consumo[37:182,"consumo"], lag.max = 30)
# ccf(base_consumo[37:207,"ice"], base_consumo[37:207,"consumo"], lag.max = 30)
# ccf(base_consumo[37:208,"inflacion"], base_consumo[37:208,"consumo"], lag.max = 30)
# ccf(base_consumo[37:208,"m2"], base_consumo[37:208,"consumo"], lag.max = 30)
# ccf(base_consumo[37:208,"roe_sf"], base_consumo[37:208,"consumo"], lag.max = 30)
# ccf(base_consumo[37:208,"precio_wti"], base_consumo[37:208,"consumo"], lag.max = 30)
# ccf(base_consumo[37:208,"tasa_activa"], base_consumo[37:208,"consumo"], lag.max = 30)
# ccf(base_consumo[37:208,"tasa_pasiva"], base_consumo[37:208,"consumo"], lag.max = 30)




# New features ------------------------------------------------------------

## Lagged features after the ccf and acp analysis 

base_consumo <- base_consumo %>% mutate(consumo_l1 = lag(consumo, n = 1L),
                                        consumo_l2 = lag(consumo, n = 2L),
                                        consumo_l3 = lag(consumo, n = 3L),
                                        pichincha_l5 = lag(credito_banco_pichincha_ecuador, n = 5L),
                                        simulador_l4 = lag(simulador_de_credito_ecuador, n = 4L),
                                        simulador_l5 = lag(simulador_de_credito_ecuador, n = 5L),
                                        simulador_l6 = lag(simulador_de_credito_ecuador, n = 6L),
                                        credito_quirografario_l2 = lag(credito_quirografario_ecuador, n = 2L),
                                        credito_quirografario_l3 = lag(credito_quirografario_ecuador, n = 3L),
                                        prestamo_quirografario_l2 = lag(prestamo_quirografario_ecuador, n = 2L),
                                        prestamo_quirografario_l3 = lag(prestamo_quirografario_ecuador, n = 3L),
                                        guayaquil_l6 = lag(credito_banco_guayaquil_ecuador, n = 6L),
                                        credito_l5 = lag(credito_ecuador, n = 5L),
                                        credito_l6 = lag(credito_ecuador, n = 6L),
                                        prestamo_l5 = lag(prestamo_ecuador, n = 5L),
                                        prestamo_l6 = lag(prestamo_ecuador, n = 6L),
                                        inflacion_l5 = lag(inflacion, n = 5L),
                                        inflacion_l6 = lag(inflacion, n = 6L),
                                        roe_l2 = lag(roe_sf, n = 2L),
                                        roe_l3 = lag(roe_sf, n = 3L),
                                        tasa_pasiva_l1 = lag(tasa_pasiva, n = 1L),
                                        )

base_hogares <- base_hogares %>% mutate(credito_hogares_l1 = lag(credito_hogares, n = 1L),
                                        credito_hogares_l2 = lag(credito_hogares, n = 2L),
                                        credito_hogares_l3 = lag(credito_hogares, n = 3L),
                                        pichincha_l5 = lag(credito_banco_pichincha_ecuador, n = 5L),
                                        simulador_l4 = lag(simulador_de_credito_ecuador, n = 4L),
                                        simulador_l5 = lag(simulador_de_credito_ecuador, n = 5L),
                                        simulador_l6 = lag(simulador_de_credito_ecuador, n = 6L),
                                        credito_quirografario_l2 = lag(credito_quirografario_ecuador, n = 2L),
                                        credito_quirografario_l3 = lag(credito_quirografario_ecuador, n = 3L),
                                        prestamo_quirografario_l2 = lag(prestamo_quirografario_ecuador, n = 2L),
                                        prestamo_quirografario_l3 = lag(prestamo_quirografario_ecuador, n = 3L),
                                        guayaquil_l6 = lag(credito_banco_guayaquil_ecuador, n = 6L),
                                        credito_l5 = lag(credito_ecuador, n = 5L),
                                        credito_l6 = lag(credito_ecuador, n = 6L),
                                        prestamo_l5 = lag(prestamo_ecuador, n = 5L),
                                        prestamo_l6 = lag(prestamo_ecuador, n = 6L),
                                        inflacion_l5 = lag(inflacion, n = 5L),
                                        inflacion_l6 = lag(inflacion, n = 6L),
                                        roe_l2 = lag(roe_sf, n = 2L),
                                        roe_l3 = lag(roe_sf, n = 3L),
                                        precio_wti_l1 = lag(precio_wti, n = 1L),
                                        tasa_pasiva_l1 = lag(tasa_pasiva, n = 1L)
                                        ) 


## Features subset

# base_hogares <- base_hogares[37:209,c(1:2,21:44)]

# Save Data ---------------------------------------------------------------

# base_hipotecario %>% as.data.frame() %>% write.xlsx("base_hipotecario.xlsx",row.names = F, showNA = F)
# base_consumo %>% as.data.frame() %>% write.xlsx("/Users/Matt/PycharmProjects/credit_volume_forecast/base_consumo.xlsx",row.names = F, showNA = F)
base_hogares %>% as.data.frame() %>% write.xlsx("/Users/Matt/PycharmProjects/credit_volume_forecast/base_hogares.xlsx",row.names = F, showNA = F)




