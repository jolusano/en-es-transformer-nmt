### Worst 20 translations — en-es

| # | sBLEU | Source | Reference | Model output | Failure modes |
|---|-------|--------|-----------|--------------|---------------|
| 1 | 0.0 | Everyone escaped. | Escaparon todos. | Todos escaparon. | other |
| 2 | 0.0 | See you later. | Nos estamos viendo. | Hasta luego. | no_content_overlap |
| 3 | 0.0 | I am Japanese. | Soy japonesa. | Soy japonés. | other |
| 4 | 0.0 | I hate it. | Me la seca. | Lo odio. | no_content_overlap |
| 5 | 0.0 | I followed him. | Le seguí. | Le seguí. | other |
| 6 | 0.0 | When do we go? | ¿Cuándo vamos? | ¿Cuándo vamos? | other |
| 7 | 0.0 | Does she like me? | ¿Le gusto a ella? | ¿Le gusto? | other |
| 8 | 0.0 | I am Japanese. | Soy japonés. | Soy japonés. | other |
| 9 | 0.0 | I learn fast. | Aprendo rápido. | Aprendo rápido. | other |
| 10 | 0.0 | I want to come in. | Quiero entrar. | Quiero entrar. | other |
| 11 | 0.0 | I have gained weight. | He ganado peso. | He engordado. | no_content_overlap |
| 12 | 0.0 | I need answers. | Necesito respuestas. | Necesito respuestas. | other |
| 13 | 0.0 | Tom waited. | Tomás esperó. | Tom esperó. | other |
| 14 | 0.0 | I hate it. | Me la baja. | Lo odio. | no_content_overlap |
| 15 | 0.0 | It's July. | Ahora es julio. | Es julio. | other |
| 16 | 0.0 | Did you find anything? | ¿Descubriste algo? | ¿Encontraste algo? | other |
| 17 | 0.0 | Are you beginners? | ¿Sois principiantes? | ¿Sois principiantes? | other |
| 18 | 0.0 | I bought one. | Compré uno. | Compré una. | other |
| 19 | 0.0 | Do you have family? | ¿Tienes familia? | ¿Tienes familia? | other |
| 20 | 0.0 | We continued walking. | Continuamos andando. | Seguimos caminando. | no_content_overlap |

**Failure-mode frequency across the whole test set**

| Category | Sentences | Rate |
|----------|-----------|------|
| other | 4362 | 77.1% |
| unknown_token | 1119 | 19.8% |
| no_content_overlap | 243 | 4.3% |
| repetition | 208 | 3.7% |
| over_generation | 106 | 1.9% |
| number_mismatch | 14 | 0.2% |
| truncation | 12 | 0.2% |
| copied_source | 8 | 0.1% |


### Worst 20 translations — es-en

| # | sBLEU | Source | Reference | Model output | Failure modes |
|---|-------|--------|-----------|--------------|---------------|
| 1 | 0.0 | ¡Despacio! | Slow down! | Slow down! | other |
| 2 | 0.0 | Nos estamos viendo. | See you later. | We're watching. | no_content_overlap |
| 3 | 0.0 | Soy japonesa. | I am Japanese. | I'm Japanese. | other |
| 4 | 0.0 | Somos confiables. | We're credible. | We're superstitious. | other |
| 5 | 0.0 | Los políticos mienten. | Politicians lie. | Politicians lie. | other |
| 6 | 0.0 | Soy japonés. | I am Japanese. | I'm Japanese. | other |
| 7 | 0.0 | Gracias de todas formas. | Thanks all the same. | Thanks anyway. | other |
| 8 | 0.0 | Él es veloz. | He's fast. | He's fast. | other |
| 9 | 0.0 | Ella es tranquila. | She is quiet. | She's quiet. | other |
| 10 | 0.0 | Ven a casa. | Come home. | Come home. | other |
| 11 | 0.0 | Tomás esperó. | Tom waited. | Tom waited. | other |
| 12 | 0.0 | Él es americano. | He is an American. | He's American. | other |
| 13 | 0.0 | Estupendo. | I stayed up late. | great. | truncation, no_content_overlap |
| 14 | 0.0 | Él es americano. | He is American. | He's American. | other |
| 15 | 0.0 | Valía la pena probar. | It was worth trying. | worth trying. | other |
| 16 | 0.0 | Elige sabiamente. | Choose carefully. | Choose wisely. | other |
| 17 | 0.0 | Continuamos andando. | We continued walking. | Keep walking. | other |
| 18 | 0.0 | ¿Quién se está riendo? | Who's laughing? | Who's laughing? | other |
| 19 | 0.0 | Somos pacientes. | We're patients. | We're patients. | other |
| 20 | 0.0 | Gracias de todas formas. | Thanks anyway. | Thanks anyway. | other |

**Failure-mode frequency across the whole test set**

| Category | Sentences | Rate |
|----------|-----------|------|
| other | 4781 | 84.5% |
| unknown_token | 701 | 12.4% |
| no_content_overlap | 167 | 3.0% |
| repetition | 138 | 2.4% |
| over_generation | 45 | 0.8% |
| number_mismatch | 25 | 0.4% |
| truncation | 11 | 0.2% |
| copied_source | 4 | 0.1% |
