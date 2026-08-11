### Worst 20 translations — en-es

| # | sBLEU | Source | Reference | Model output | Failure modes |
|---|-------|--------|-----------|--------------|---------------|
| 1 | 0.0 | Slow down! | ¡Despacio! | ¡Qué despacio! | no_content_overlap |
| 2 | 0.0 | I am Japanese. | Soy japonesa. | Soy japonés. | other |
| 3 | 0.0 | I hate it. | Me la seca. | Lo odio. | no_content_overlap |
| 4 | 0.0 | I followed him. | Le seguí. | Le seguí. | other |
| 5 | 0.0 | We're credible. | Somos confiables. | Somos creíbles. | other |
| 6 | 0.0 | I belched. | Eructé. | Me dolía. | no_content_overlap |
| 7 | 0.0 | I'll dance. | Bailaré. | Bailaré. | other |
| 8 | 0.0 | I am Japanese. | Soy japonés. | Soy japonés. | other |
| 9 | 0.0 | Surprise me. | Sorprendedme. | Me sorprende. | no_content_overlap |
| 10 | 0.0 | I learn fast. | Aprendo rápido. | Aprendo rápido. | other |
| 11 | 0.0 | I want to come in. | Quiero entrar. | Quiero entrar. | other |
| 12 | 0.0 | I have gained weight. | He ganado peso. | He engordado. | no_content_overlap |
| 13 | 0.0 | I need answers. | Necesito respuestas. | Necesito respuestas. | other |
| 14 | 0.0 | Tom waited. | Tomás esperó. | Tom esperó. | other |
| 15 | 0.0 | I hate it. | Me la baja. | Lo odio. | no_content_overlap |
| 16 | 0.0 | It's July. | Ahora es julio. | Es julio. | other |
| 17 | 0.0 | Did you find anything? | ¿Descubriste algo? | ¿Encontraste algo? | other |
| 18 | 0.0 | Are you beginners? | ¿Sois principiantes? | ¿Eres principiante? | no_content_overlap |
| 19 | 0.0 | They need instructions. | Ellos necesitan instrucciones. | Necesitan instrucciones. | other |
| 20 | 0.0 | I bought one. | Compré uno. | Compré uno. | other |

**Failure-mode frequency across the whole test set**

| Category | Sentences | Rate |
|----------|-----------|------|
| other | 5357 | 94.7% |
| no_content_overlap | 215 | 3.8% |
| repetition | 33 | 0.6% |
| truncation | 24 | 0.4% |
| over_generation | 22 | 0.4% |
| number_mismatch | 12 | 0.2% |
| copied_source | 11 | 0.2% |


### Worst 20 translations — es-en

| # | sBLEU | Source | Reference | Model output | Failure modes |
|---|-------|--------|-----------|--------------|---------------|
| 1 | 0.0 | Escaparon todos. | Everyone escaped. | Everyone escaped. | other |
| 2 | 0.0 | Gracias de todos modos. | Thank you just the same. | Thanks anyway. | truncation, no_content_overlap |
| 3 | 0.0 | ¡Despacio! | Slow down! | Easy down! | other |
| 4 | 0.0 | Soy japonesa. | I am Japanese. | I'm Japanese. | other |
| 5 | 0.0 | Me la seca. | I hate it. | My dry. | no_content_overlap |
| 6 | 0.0 | Somos confiables. | We're credible. | We're confiable. | other |
| 7 | 0.0 | Eructé. | I belched. | I burped. | no_content_overlap |
| 8 | 0.0 | Bailaré. | I'll dance. | I'll dance. | other |
| 9 | 0.0 | Los políticos mienten. | Politicians lie. | Politicians lie. | other |
| 10 | 0.0 | Soy japonés. | I am Japanese. | I'm Japanese. | other |
| 11 | 0.0 | Él es veloz. | He's fast. | He's fast. | other |
| 12 | 0.0 | Ella es tranquila. | She is quiet. | She's quiet. | other |
| 13 | 0.0 | Ven a casa. | Come home. | Come home. | other |
| 14 | 0.0 | Tomás esperó. | Tom waited. | Tom waited. | other |
| 15 | 0.0 | Me la baja. | I hate it. | I'm short. | no_content_overlap |
| 16 | 0.0 | Estupendo. | I stayed up late. | I stretch. | no_content_overlap |
| 17 | 0.0 | Elige sabiamente. | Choose carefully. | Choose wisely. | other |
| 18 | 0.0 | Vitorearon. | They cheered. | They dressed. | no_content_overlap |
| 19 | 0.0 | ¿Quién se está riendo? | Who's laughing? | Who's laughing? | other |
| 20 | 0.0 | Somos pacientes. | We're patients. | We're patients. | other |

**Failure-mode frequency across the whole test set**

| Category | Sentences | Rate |
|----------|-----------|------|
| other | 5449 | 96.3% |
| no_content_overlap | 129 | 2.3% |
| repetition | 33 | 0.6% |
| number_mismatch | 21 | 0.4% |
| truncation | 20 | 0.4% |
| over_generation | 15 | 0.3% |
| copied_source | 7 | 0.1% |
