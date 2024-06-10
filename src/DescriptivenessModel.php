<?php

namespace Taheralfayad\IsItDescriptive;

class DescriptivenessModel {
    public function isDescriptive($text) {
        $escaped_text = escapeshellarg($text);
        $path_to_model = realpath(__DIR__ . '/model.py');
        return shell_exec("python3 $path_to_model $escaped_text");
    }
}
