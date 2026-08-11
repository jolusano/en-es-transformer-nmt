### Worst 20 translations — en-es

| # | sBLEU | Source | Reference | Model output | Failure modes |
|---|-------|--------|-----------|--------------|---------------|
| 1 | 0.0 | See you later. | Nos estamos viendo. | veo tarde. | no_content_overlap |
| 2 | 0.0 | I am Japanese. | Soy japonesa. | Soy japonés. | other |
| 3 | 0.0 | I hate it. | Me la seca. | Odio. | truncation, no_content_overlap |
| 4 | 0.0 | We're credible. | Somos confiables. | somos credible. | other |
| 5 | 0.0 | When do we go? | ¿Cuándo vamos? | ¿Cuándo Vamos? | other |
| 6 | 0.0 | It happens to me. | A mí me sucede. | me sucede. | other |
| 7 | 0.0 | You might need it. | Podrías necesitarlo. | podría hacerlo. | no_content_overlap |
| 8 | 0.0 | I am Japanese. | Soy japonés. | Soy japonés. | other |
| 9 | 0.0 | You're driving slowly. | Estás conduciendo despacio. | Eres rápido. | no_content_overlap |
| 10 | 0.0 | I learn fast. | Aprendo rápido. | aprendiendo rápido. | other |
| 11 | 0.0 | You must die. | Vos debés morir. | Debes morir. | other |
| 12 | 0.0 | I hate horror films. | Odio las películas de horror. | Odio películas. | truncation |
| 13 | 0.0 | It has worked out nicely. | Salió de maravillas. | ha trabajado. | no_content_overlap |
| 14 | 0.0 | I have gained weight. | He ganado peso. | Tengo peso. | other |
| 15 | 0.0 | I need answers. | Necesito respuestas. | Necesito respuestas. | other |
| 16 | 0.0 | Tom waited. | Tomás esperó. | Tom esperó. | other |
| 17 | 0.0 | I hate it. | Me la baja. | Odio. | truncation, no_content_overlap |
| 18 | 0.0 | It's July. | Ahora es julio. | es junio. | no_content_overlap |
| 19 | 0.0 | Are you beginners? | ¿Sois principiantes? | ¿Eres principiantes? | other |
| 20 | 0.0 | What a big dog! | ¡Qué perro tan grande! | ¡qué perro! | other |

**Failure-mode frequency across the whole test set**

| Category | Sentences | Rate |
|----------|-----------|------|
| other | 3395 | 60.0% |
| unknown_token | 1792 | 31.7% |
| no_content_overlap | 610 | 10.8% |
| repetition | 550 | 9.7% |
| over_generation | 166 | 2.9% |
| truncation | 59 | 1.0% |
| copied_source | 47 | 0.8% |
| number_mismatch | 34 | 0.6% |


### Worst 20 translations — es-en

| # | sBLEU | Source | Reference | Model output | Failure modes |
|---|-------|--------|-----------|--------------|---------------|
| 1 | 0.0 | ¡Despacio! | Slow down! | slow! | other |
| 2 | 0.0 | Soy japonesa. | I am Japanese. | I'm Japanese. | other |
| 3 | 0.0 | Eso es mentira. | That's a lie. | That's true. | other |
| 4 | 0.0 | Soy japonés. | I am Japanese. | I'm Japanese. | other |
| 5 | 0.0 | Estás conduciendo despacio. | You're driving slowly. | You're speeding. | other |
| 6 | 0.0 | Ven a casa. | Come home. | come home. | other |
| 7 | 0.0 | Tomás esperó. | Tom waited. | Tom waited. | other |
| 8 | 0.0 | Ahora es julio. | It's July. | It's today. | other |
| 9 | 0.0 | Él es americano. | He is an American. | He's American. | other |
| 10 | 0.0 | Estupendo. | I stayed up late. | estupendo. | truncation, copied_source, no_content_overlap |
| 11 | 0.0 | Él es americano. | He is American. | He's American. | other |
| 12 | 0.0 | Somos pacientes. | We're patients. | We're pacientes. | other |
| 13 | 0.0 | Bésame. | Kiss me. | I love. | no_content_overlap |
| 14 | 0.0 | ¡Sentaos! | Sit down! | give me! | no_content_overlap |
| 15 | 0.0 | Eres terco. | You're stubborn. | You're terco. | other |
| 16 | 0.0 | ¿A quién amas? | Who do you love? | who loves? | other |
| 17 | 0.0 | Nadie sabe. | Nobody knows. | Nobody knows. | other |
| 18 | 0.0 | Están malos. | They're bad. | They're malos. | other |
| 19 | 0.0 | He comprendido. | I understood. | I've completed. | no_content_overlap |
| 20 | 0.0 | Los domingos descanso. | On Sundays I rest. | Sundays. | truncation |

**Failure-mode frequency across the whole test set**

| Category | Sentences | Rate |
|----------|-----------|------|
| other | 3927 | 69.4% |
| unknown_token | 1020 | 18.0% |
| no_content_overlap | 589 | 10.4% |
| repetition | 452 | 8.0% |
| copied_source | 145 | 2.6% |
| over_generation | 98 | 1.7% |
| truncation | 51 | 0.9% |
| number_mismatch | 48 | 0.8% |
